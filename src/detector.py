

import warnings, traceback
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from scipy.sparse import hstack

warnings.filterwarnings('ignore')

# Optional libs
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    HAS_ST = False

try:
    import textstat
    HAS_TS = True
except Exception:
    HAS_TS = False

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

print(f"Libraries loaded. sentence-transformers={HAS_ST}, textstat={HAS_TS}")

# ======================================================
# Feature Extractor
# ======================================================
class ChampionFeatureExtractor:
    def __init__(self, use_embeddings=False, embed_model='all-MiniLM-L6-v2'):
        self.use_embeddings = use_embeddings and HAS_ST
        if self.use_embeddings:
            try:
                self.embedder = SentenceTransformer(embed_model)
            except Exception:
                print("⚠️  SentenceTransformer failed to load — embeddings disabled.")
                self.use_embeddings = False

    def _basic(self, s: pd.Series) -> pd.DataFrame:
        s = s.fillna('').astype(str)
        text_len = s.str.len()
        word_ct = s.str.split().str.len()
        sent_ct = s.str.count(r'[.!?]+')
        return pd.DataFrame({
            'text_len': text_len,
            'word_ct': word_ct,
            'avg_word_len': text_len/(word_ct+1),
            'sent_ct': sent_ct,
            'avg_sent_len': word_ct/(sent_ct+1),
            'commas': s.str.count(','),
            'periods': s.str.count(r'\.'),
        })

    def _readability(self, s: pd.Series) -> pd.DataFrame:
        if not HAS_TS:
            return pd.DataFrame(np.zeros((len(s),4)),columns=['flesch','fog','smog','coleman'])
        s = s.fillna('')
        return pd.DataFrame({
            'flesch':  s.apply(lambda x: textstat.flesch_reading_ease(x) if x else 0),
            'fog':     s.apply(lambda x: textstat.gunning_fog(x)        if x else 0),
            'smog':    s.apply(lambda x: textstat.smog_index(x)         if x else 0),
            'coleman': s.apply(lambda x: textstat.coleman_liau_index(x) if x else 0)
        })

    def extract_embeddings(self, s: pd.Series):
        if not self.use_embeddings: return None
        try:
            return np.asarray(self.embedder.encode(s.fillna('').tolist(), show_progress_bar=False))
        except Exception as e:
            print("Embedding extraction failed:", e)
            return None

    def extract_features(self, df: pd.DataFrame):
        s = df['answer']
        basic = self._basic(s)
        read  = self._readability(s)
        feat  = pd.concat([basic, read], axis=1)
        if 'topic' in df.columns:
            feat = pd.concat([feat, pd.get_dummies(df['topic'].fillna(''), prefix='topic')], axis=1)
        emb = self.extract_embeddings(s)
        return feat, emb


# ======================================================
# Main Detector
# ======================================================
class ChampionDetector:
    def __init__(self, use_embeddings=False, svd_components=256, random_state=42):
        self.fe = ChampionFeatureExtractor(use_embeddings)
        self.svd_components = svd_components
        self.random_state = random_state
        self.meta_model = None
        self.oof_pred = None

    def _fit_vec(self, texts: pd.Series):
        tfw = TfidfVectorizer(max_features=3000, ngram_range=(1,3),
                              stop_words='english', sublinear_tf=True, min_df=2, max_df=0.9)
        tfc = TfidfVectorizer(max_features=1500, analyzer='char_wb',
                              ngram_range=(3,5), sublinear_tf=True, min_df=2)
        Xw = tfw.fit_transform(texts)
        Xc = tfc.fit_transform(texts)
        svd = TruncatedSVD(n_components=min(self.svd_components, Xw.shape[1]+Xc.shape[1]-1),
                           random_state=self.random_state)
        svd.fit(hstack([Xw, Xc], format='csr'))
        return {'tfw': tfw, 'tfc': tfc, 'svd': svd}

    def _tr_vec(self, texts: pd.Series, v):
        Xw = v['tfw'].transform(texts)
        Xc = v['tfc'].transform(texts)
        return v['svd'].transform(hstack([Xw, Xc], format='csr'))

    # -------- per-fold (robust) ------------
    def _prepare_fold(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        # Build ONLY on the fold subsets; reset_index to keep lengths clean
        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)

        train_f, train_e = self.fe.extract_features(train_df)
        val_f,   val_e   = self.fe.extract_features(val_df)

        # Align handcrafted columns
        val_f = val_f.reindex(columns=train_f.columns, fill_value=0)

        # Fit vectorizers+SVD on train fold; transform both
        v      = self._fit_vec(train_df['answer'])
        Xt_tr  = self._tr_vec(train_df['answer'], v)
        Xt_val = self._tr_vec(val_df['answer'],   v)

        Xtr_parts = [Xt_tr]
        Xva_parts = [Xt_val]

        # embeddings (safe handling)
        if (train_e is not None) and (len(train_e) == len(train_df)):
            Xtr_parts.append(train_e)
            if (val_e is not None) and (len(val_e) == len(val_df)):
                Xva_parts.append(val_e)
            else:
                Xva_parts.append(np.zeros((len(val_df), train_e.shape[1])))

        # handcrafted numeric
        Xtr_parts.append(train_f.values)
        Xva_parts.append(val_f.values)

        # shape guards (truncate to the shortest length if anything drifts)
        lens_tr = [x.shape[0] for x in Xtr_parts]
        if len(set(lens_tr)) != 1:
            print("⚠️ Train part mismatch:", lens_tr)
            m = min(lens_tr); Xtr_parts = [x[:m] for x in Xtr_parts]
        lens_va = [x.shape[0] for x in Xva_parts]
        if len(set(lens_va)) != 1:
            print("⚠️ Val part mismatch:", lens_va)
            m = min(lens_va); Xva_parts = [x[:m] for x in Xva_parts]

        Xtr = np.hstack(Xtr_parts)
        Xva = np.hstack(Xva_parts)
        hand_cols = train_f.columns
        return Xtr, Xva, hand_cols

    # -------- full-train (robust) ----------
    def _prepare_full(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        trf, tre = self.fe.extract_features(train_df)
        tef, tee = self.fe.extract_features(test_df)
        tef = tef.reindex(columns=trf.columns, fill_value=0)

        v      = self._fit_vec(train_df['answer'])
        Xt_tr  = self._tr_vec(train_df['answer'], v)
        Xt_te  = self._tr_vec(test_df['answer'],   v)

        Xtr_parts = [Xt_tr]
        Xte_parts = [Xt_te]

        if (tre is not None) and (len(tre) == len(train_df)):
            Xtr_parts.append(tre)
            if (tee is not None) and (len(tee) == len(test_df)):
                Xte_parts.append(tee)
            else:
                Xte_parts.append(np.zeros((len(test_df), tre.shape[1])))

        Xtr_parts.append(trf.values)
        Xte_parts.append(tef.values)

        # guards
        lens_tr = [x.shape[0] for x in Xtr_parts]
        if len(set(lens_tr)) != 1:
            print("⚠️ FULL Train part mismatch:", lens_tr)
            m = min(lens_tr); Xtr_parts = [x[:m] for x in Xtr_parts]
        lens_te = [x.shape[0] for x in Xte_parts]
        if len(set(lens_te)) != 1:
            print("⚠️ FULL Test part mismatch:", lens_te)
            m = min(lens_te); Xte_parts = [x[:m] for x in Xte_parts]

        Xtr = np.hstack(Xtr_parts)
        Xte = np.hstack(Xte_parts)
        return Xtr, Xte, trf.columns

    # -------------- training ---------------
    def train_ensemble(self, train_df: pd.DataFrame, n_splits=10, n_repeats=1, random_state=42):
        y = train_df['is_cheating'].values
        rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        total_folds = n_splits * n_repeats

        oof = {k: np.zeros(len(train_df)) for k in ['lgb', 'xgb', 'cat', 'lr']}

        for i, (tr, va) in enumerate(rkf.split(np.zeros(len(y)), y), 1):
            repeat_idx = (i-1)//n_splits + 1
            print(f"\nFold {i}/{total_folds} (Repeat {repeat_idx}/{n_repeats})")

            Xtr, Xv, cols = self._prepare_fold(train_df.iloc[tr], train_df.iloc[va])

            n_hand = len(cols)
            if n_hand > 0:
                sc = StandardScaler()
                Xtr[:, -n_hand:] = sc.fit_transform(Xtr[:, -n_hand:])
                Xv[:,  -n_hand:] = sc.transform(Xv[:,  -n_hand:])

            ytr, yv = y[tr], y[va]

            # LGBM
            l = lgb.LGBMClassifier(n_estimators=1200, learning_rate=0.01,
                                   max_depth=7, num_leaves=63, verbose=-1,
                                   subsample=0.8, colsample_bytree=0.7)
            l.fit(Xtr, ytr, eval_set=[(Xv, yv)],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            oof['lgb'][va] = l.predict_proba(Xv)[:, 1]

            # XGB
            x = xgb.XGBClassifier(n_estimators=1200, learning_rate=0.01, max_depth=6,
                                  subsample=0.8, colsample_bytree=0.7, tree_method='hist',
                                  eval_metric='auc', use_label_encoder=False)
            x.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
            oof['xgb'][va] = x.predict_proba(Xv)[:, 1]

            # CatBoost
            c = CatBoostClassifier(iterations=800, learning_rate=0.02, depth=6,
                                   verbose=0, early_stopping_rounds=100)
            c.fit(Xtr, ytr, eval_set=(Xv, yv), verbose=False)
            oof['cat'][va] = c.predict_proba(Xv)[:, 1]

            # Logistic Regression (on fully scaled block)
            sc2 = StandardScaler()
            Xtr_s = sc2.fit_transform(Xtr)
            Xv_s  = sc2.transform(Xv)
            lr = LogisticRegression(C=0.3, max_iter=2000, solver='saga', n_jobs=-1)
            lr.fit(Xtr_s, ytr)
            oof['lr'][va] = lr.predict_proba(Xv_s)[:, 1]

            fold_auc = roc_auc_score(yv, (oof['lgb'][va] + oof['xgb'][va] + oof['cat'][va] + oof['lr'][va]) / 4.0)
            print("Fold AUC:", round(fold_auc, 6))

        # Meta-stacking
        meta = np.vstack([oof[k] for k in oof]).T
        self.meta_model = LogisticRegressionCV(cv=5, scoring='roc_auc', max_iter=2000).fit(meta, y)
        self.oof_pred   = self.meta_model.predict_proba(meta)[:, 1]
        auc = roc_auc_score(y, self.oof_pred)
        print("\nMeta OOF AUC:", round(auc, 8))
        return auc

    # -------------- inference --------------
    def predict(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        Xtr, Xte, cols = self._prepare_full(train_df, test_df)
        y = train_df['is_cheating'].values

        # scale handcrafted
        n_hand = len(cols)
        if n_hand > 0:
            sc = StandardScaler()
            Xtr[:, -n_hand:] = sc.fit_transform(Xtr[:, -n_hand:])
            Xte[:, -n_hand:] = sc.transform(Xte[:, -n_hand:])

        # base models on full data
        l = lgb.LGBMClassifier(n_estimators=1200, learning_rate=0.01, max_depth=7, num_leaves=63, verbose=-1,
                               subsample=0.8, colsample_bytree=0.7)
        l.fit(Xtr, y); lp = l.predict_proba(Xte)[:, 1]

        x = xgb.XGBClassifier(n_estimators=1200, learning_rate=0.01, max_depth=6, subsample=0.8,
                              colsample_bytree=0.7, tree_method='hist', eval_metric='auc', use_label_encoder=False)
        x.fit(Xtr, y, verbose=False); xp = x.predict_proba(Xte)[:, 1]

        c = CatBoostClassifier(iterations=800, learning_rate=0.02, depth=6, verbose=0)
        c.fit(Xtr, y); cp = c.predict_proba(Xte)[:, 1]

        sc2 = StandardScaler()
        Xtr_s = sc2.fit_transform(Xtr)
        Xte_s = sc2.transform(Xte)
        lr = LogisticRegression(C=0.3, max_iter=2000, solver='saga')
        lr.fit(Xtr_s, y); rp = lr.predict_proba(Xte_s)[:, 1]

        meta_in = np.vstack([lp, xp, cp, rp]).T
        out = self.meta_model.predict_proba(meta_in)[:, 1]

        # calibration on OOF → apply to test
        cal = IsotonicRegression(out_of_bounds='clip').fit(self.oof_pred, y)
        pred = np.clip(cal.transform(out), 0.001, 0.999)
        return pred


# ======================================================
def champion_main(n_splits=10, n_repeats=1):
    print("="*70)
    print("Champion++ v4 – Mercor AI Detection")
    print("="*70)
    try:
        train = pd.read_csv('/kaggle/input/mercor-ai-detection/train.csv')
        test  = pd.read_csv('/kaggle/input/mercor-ai-detection/test.csv')
        train['answer'] = train['answer'].fillna('')
        test['answer']  = test['answer'].fillna('')
        print(f"Train {train.shape}, Test {test.shape}, Cheating ratio {train['is_cheating'].mean():.3f}")

        det = ChampionDetector(use_embeddings=True, svd_components=256, random_state=42)
        auc = det.train_ensemble(train, n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        preds = det.predict(train, test)

        sub = pd.DataFrame({'id': test['id'], 'is_cheating': preds})
        sub.to_csv('championpp_v4.csv', index=False, float_format='%.10f')
        print(f"\nSaved championpp_v4.csv | CV AUC: {auc:.8f}")
        return sub, auc
    except Exception as e:
        print("Error:", e); traceback.print_exc()

if __name__ == "__main__":

    champion_main(n_splits=10, n_repeats=2)

