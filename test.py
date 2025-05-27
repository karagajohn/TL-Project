import streamlit as st
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Bioinformatics ML App", layout="wide")
st.title("\U0001F52C Μοριακή Βιολογία & Μηχανική Μάθηση")
st.markdown("Ανάλυση δεδομένων και πρόβλεψη μέσω Μηχανικής Μάθησης")

uploaded_file = st.file_uploader("\U0001F4C1 Ανέβασε αρχείο CSV ή .h5ad με μοριακά δεδομένα", type=["csv", "h5ad"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1]

    if file_type == "csv":
        data = pd.read_csv(uploaded_file)
        st.subheader("\U0001F4C4 CSV Δεδομένα")
        st.write(data.head())

        st.sidebar.header("⚙️ Ρυθμίσεις (CSV)")
        target_col = st.sidebar.selectbox("\U0001F3AF Επιλέξτε τη μεταβλητή-στόχο", data.columns)
        test_size = st.sidebar.slider("\U0001F4CA Μέγεθος test set (%)", 10, 50, 20)
        features = st.multiselect("\U0001F4CC Επιλογή χαρακτηριστικών", [col for col in data.columns if col != target_col])

        if features:
            X = data[features]
            y = data[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("\U0001F4C8 Αποτελέσματα")
            st.code(classification_report(y_test, y_pred), language="text")

            st.subheader("\U0001F50D Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

    elif file_type == "h5ad":
        adata = sc.read_h5ad(uploaded_file)
        st.subheader("\U0001F9EC AnnData (.h5ad) Δεδομένα")
        st.write(f"\U0001F4CA Σχήμα: {adata.shape}")
        st.write("\U0001F520 Πρώτα 10 χαρακτηριστικά:", list(adata.var_names[:10]))
        st.write("\U0001F9EC Παρατηρήσεις (obs):")
        st.write(adata.obs.head())

        obs_columns = adata.obs.columns.tolist()
        if "disease" not in obs_columns:
            st.warning("⚠️ Η στήλη 'disease' δεν υπάρχει στο adata.obs. Χρειάζεται για DEG.")
        else:
            color_by = st.selectbox("\U0001F3A8 Χρωματισμός UMAP κατά:", obs_columns)
            use_harmony = st.checkbox("🔄 Χρήση Harmony για διόρθωση batch effects")

            if st.button("\U0001F504 Εκτέλεση Preprocessing & UMAP + DEG Analysis"):
                with st.spinner("\U0001F52C Εκτέλεση προεπεξεργασίας..."):
                    sc.pp.filter_cells(adata, min_genes=600)
                    sc.pp.filter_genes(adata, min_cells=3)
                    adata = adata[:, [gene for gene in adata.var_names if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
                    adata.raw = adata
                    adata = adata[:, adata.var.highly_variable]
                    sc.pp.scale(adata, max_value=10)
                    sc.pp.pca(adata)
                    if use_harmony:
                        import scanpy.external as sce
                        sce.pp.harmony_integrate(adata, 'batch')
                        sc.pp.neighbors(adata, use_rep="X_pca_harmony")
                    else:
                        sc.pp.neighbors(adata)
                    sc.tl.umap(adata)
                    
                    st.success("✅ UMAP & preprocessing ολοκληρώθηκαν!")

                    st.subheader("\U0001F4CA UMAP Projection")
                    sc.pl.umap(adata, color=color_by, show=False)
                    st.pyplot()

                    # DEG Analysis
                    st.subheader("\U0001F4C9 DEG Analysis & Volcano Plot")
                    sc.tl.rank_genes_groups(
                        adata,
                        groupby='disease',
                        method='wilcoxon',
                        groups=['case'],
                        reference='control',
                        use_raw=False
                    )
                    deg_result = adata.uns["rank_genes_groups"]

                    degs_df = pd.DataFrame({
                        "genes": deg_result["names"]["case"],
                        "pvals": deg_result["pvals"]["case"],
                        "pvals_adj": deg_result["pvals_adj"]["case"],
                        "logfoldchanges": deg_result["logfoldchanges"]["case"],
                    })

                    degs_df["neg_log10_pval"] = -np.log10(degs_df["pvals"])
                    degs_df["diffexpressed"] = "NS"
                    degs_df.loc[(degs_df["logfoldchanges"] > 1) & (degs_df["pvals"] < 0.05), "diffexpressed"] = "UP"
                    degs_df.loc[(degs_df["logfoldchanges"] < -1) & (degs_df["pvals"] < 0.05), "diffexpressed"] = "DOWN"

                    top_downregulated = degs_df[degs_df["diffexpressed"] == "DOWN"].sort_values(by=["neg_log10_pval", "logfoldchanges"], ascending=[False, True]).head(20)
                    top_downregulated = top_downregulated.sort_values(by=["neg_log10_pval", "logfoldchanges"], ascending=[False, True]).head(20)

                    top_upregulated = degs_df[degs_df["diffexpressed"] == "UP"].sort_values(by=["neg_log10_pval", "logfoldchanges"], ascending=[False, False]).head(20)
                    top_upregulated = top_upregulated.sort_values(by=["neg_log10_pval", "logfoldchanges"], ascending=[False, False]).head(81)

                    top_genes_combined = pd.concat([top_downregulated, top_upregulated])
                    df_annotated = degs_df[degs_df["genes"].isin(top_genes_combined)]

                    st.dataframe(top_genes_combined[["genes", "logfoldchanges", "pvals", "diffexpressed"]])

                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(
                        data=degs_df,
                        x="logfoldchanges",
                        y="neg_log10_pval",
                        hue="diffexpressed",
                        palette={"UP": "#bb0c00", "DOWN": "#00AFBB", "NS": "grey"},
                        alpha=0.7,
                        edgecolor=None
                    )
                    plt.axhline(y=-np.log10(0.05), color='gray', linestyle='dashed')
                    plt.axvline(x=-1, color='gray', linestyle='dashed')
                    plt.axvline(x=1, color='gray', linestyle='dashed')
                    plt.xlim(-11, 11)
                    plt.ylim(25, 175)
                    plt.xlabel("log2 Fold Change", fontsize=14)
                    plt.ylabel("-log10 p-value", fontsize=14)
                    plt.title("Volcano of DEGs (Case vs Control)", fontsize=16)
                    plt.legend(title="Expression", loc="upper right")
                    st.pyplot()

# About
st.sidebar.markdown("---")
st.sidebar.subheader("\U0001F468‍\U0001F4BB Ομάδα")
st.sidebar.text(
    "ΟΝΟΜΑΤΑ:\n"
    "- ΜΟΥΛΑΚΑΚΗΣ ΙΩΑΝΝΗΣ\n"
    "-inf2022122\n"
    "- ΚΑΡΑΓΚΑΔΑΚΗΣ ΙΩΑΝΝΗΣ\n"
    "-inf2021079\n"
    "- ΚΩΝΣΤΑΝΤΙΝΟΣ ΤΣΙΛΩΝΗΣ\n"
    "-inf2022217\n"
)