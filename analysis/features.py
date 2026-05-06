"""
Feature name resolution helpers for multi-omics attribution analysis.

- RNA: Ensembl ID (optionally versioned) -> HGNC gene symbol via mygene
- Methylation: CpG probe ID -> gene name via HM450 manifest CSV
- miRNA: strip "hsa-" prefix for display
"""

import pandas as pd


def build_rna_symbol_map(feature_names_rna: list) -> dict:
    """
    Resolve a list of Ensembl IDs (optionally version-suffixed) to HGNC gene symbols.

    Queries the mygene.info API in batch. Versioned IDs like "ENSG00000141510.7"
    are stripped to "ENSG00000141510" before querying.

    Args:
        feature_names_rna: list of raw Ensembl feature names (versioned or plain)

    Returns:
        dict mapping each raw ID in feature_names_rna to its symbol (or None if
        unresolved). Both versioned ("ENSG...X.Y") and unversioned ("ENSG...X")
        forms are included as keys for lookup convenience.
    """
    try:
        import mygene
    except ImportError as e:
        raise ImportError("mygene is required for RNA symbol resolution: pip install mygene") from e

    mg = mygene.MyGeneInfo()

    raw_ids   = feature_names_rna
    clean_ids = [e.split(".")[0] for e in raw_ids]

    results = mg.querymany(
        clean_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        verbose=False,
        returnall=False,
    )

    clean_to_symbol = {}
    for r in results:
        if "symbol" in r and not r.get("notfound", False):
            clean_to_symbol[r["query"]] = r["symbol"]

    symbol_map = {}
    for raw, clean in zip(raw_ids, clean_ids):
        sym = clean_to_symbol.get(clean)
        symbol_map[raw]   = sym
        symbol_map[clean] = sym

    n_resolved = sum(v is not None for k, v in symbol_map.items() if k in raw_ids)
    print(f"RNA: resolved {n_resolved} / {len(raw_ids)} Ensembl IDs to symbols")

    return symbol_map


def build_probe_gene_map(feature_names_methyl: list, manifest_path: str) -> dict:
    """
    Map CpG probe IDs to gene names using the Illumina HM450 manifest CSV.

    The manifest is expected in the standard Illumina format (7-row header,
    probe ID as the index column, "UCSC_RefGene_Name" as the gene column).
    Semicolon-delimited gene lists are reduced to the first entry.
    Intergenic probes and probes absent from the manifest are mapped to None.

    Args:
        feature_names_methyl : list of CpG probe IDs
        manifest_path        : path to the HM450 manifest CSV file

    Returns:
        dict mapping each probe ID to a gene name string or None
    """
    manifest = pd.read_csv(manifest_path, skiprows=7, index_col=0, low_memory=False)

    probe_to_gene = {}
    for pid in feature_names_methyl:
        if pid not in manifest.index:
            probe_to_gene[pid] = None
            continue
        gene = manifest.loc[pid, "UCSC_RefGene_Name"]
        if pd.notna(gene) and str(gene).strip():
            probe_to_gene[pid] = gene.split(";")[0].strip()
        else:
            probe_to_gene[pid] = None

    n_resolved = sum(v is not None for v in probe_to_gene.values())
    print(f"Methylation: resolved {n_resolved} / {len(probe_to_gene)} probes to genes")

    return probe_to_gene


def clean_mirna_name(name: str) -> str:
    """Strip the 'hsa-' prefix from a miRNA name for display."""
    return name.replace("hsa-", "")


def resolve_feature_name(
    mod:               str,
    raw:               str,
    ensembl_to_symbol: dict = None,
    probe_to_gene:     dict = None,
) -> str:
    """
    Return a human-readable display name for a single feature.

    Args:
        mod               : one of "rna", "mirna", "methyl"
        raw               : raw feature name (Ensembl ID / miRNA ID / probe ID)
        ensembl_to_symbol : output of build_rna_symbol_map(); required for RNA
        probe_to_gene     : output of build_probe_gene_map(); required for methylation

    Returns:
        display string
    """
    if mod == "rna":
        if ensembl_to_symbol is None:
            return raw.split(".")[0]
        clean = raw.split(".")[0]
        return ensembl_to_symbol.get(raw) or ensembl_to_symbol.get(clean) or clean

    if mod == "mirna":
        return clean_mirna_name(raw)

    if mod == "methyl":
        gene = (probe_to_gene or {}).get(raw)
        return f"{gene}\n({raw})" if gene else f"intergenic\n({raw})"

    return raw


def get_feature_names(dataset_cls=None) -> dict:
    """
    Build a feature names dict from MultiOmicsDataset class attributes (if loaded),
    falling back to generic index-based names.

    Args:
        dataset_cls: MultiOmicsDataset class (imported by caller to avoid circular deps)

    Returns:
        dict with keys "rna", "mirna", "methyl", each a list of feature name strings
    """
    names = {}
    for mod, attr, default_prefix in [
        ("rna",    "_rna_feature_names",    "gene"),
        ("mirna",  "_mirna_feature_names",  "mirna"),
        ("methyl", "_methyl_feature_names", "cpg"),
    ]:
        if dataset_cls is not None and hasattr(dataset_cls, attr):
            stored = getattr(dataset_cls, attr)
            if stored is not None:
                names[mod] = list(stored)
                continue
        # Fallback: generic names
        data_attr = f"_{mod}_data"
        if dataset_cls is not None and hasattr(dataset_cls, data_attr):
            arr = getattr(dataset_cls, data_attr)
            if arr is not None:
                names[mod] = [f"{default_prefix}_{i}" for i in range(arr.shape[1])]
                continue
        names[mod] = []
    return names
