from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "article" / "elsarticle_manuscript" / "references.bib"


DOI_REFS = [
    ("Zhu_2023_DynamicEnsemble", "10.1016/j.ins.2022.12.022"),
    ("Wang_2023_MultiKernel", "10.1016/j.ins.2023.119462"),
    ("Yang_2023_StableLabel", "10.1016/j.ins.2023.119525"),
    ("Hao_2024_ViewSpecific", "10.1016/j.ins.2024.121215"),
    ("Zhang_2024_LabelEnhancement", "10.1016/j.ins.2024.121113"),
    ("Liu_2024_LabelRelaxation", "10.1016/j.ins.2024.120662"),
    ("Wang_2024_ConsistentSpecific", "10.1016/j.ins.2024.121395"),
    ("Tan_2023_PrivilegedMVML", "10.1016/j.ins.2023.119911"),
    ("Gao_2024_KnowledgeConsistency", "10.1016/j.ins.2024.120870"),
    ("Sun_2023_HighOrderCorrelation", "10.1016/j.ins.2022.12.072"),
    ("Huang_2023_CostConstrained", "10.1016/j.patcog.2023.109605"),
    ("Gibaja_2023_Multidimensional", "10.1016/j.patcog.2023.109357"),
    ("Yuan_2023_MFSJMI", "10.1016/j.patcog.2023.109378"),
    ("Xu_2018_PrivilegedInfo", "10.1016/j.patcog.2018.03.033"),
    ("Huang_2023_CorrelationFS", "10.1016/j.patcog.2023.109899"),
    ("Li_2022_GenerativeCorrelation", "10.1145/3538708"),
    ("Zhang_2020_MutualInfoLD", "10.1016/j.knosys.2020.105684"),
    ("Liu_2024_PartitionLD", "10.1109/TNNLS.2023.3341807"),
    ("Zhou_2023_ExploitLD", "10.1109/TNNLS.2021.3103178"),
    ("Peng_2019_HierarchicalMLTC", "10.1145/3357384.3357885"),
    ("Banerjee_2019_TransferHTC", "10.18653/v1/P19-1633"),
    ("Chalkidis_2020_CorrelationXML", "10.1145/3394486.3403151"),
    ("Jiang_2020_TransformersXML", "10.1145/3394486.3403368"),
    ("Kowsari_2019_HierarchicalExtreme", "10.1016/j.asoc.2019.03.041"),
    ("Liu_2024_LabelAttentionHistorical", "10.1016/j.knosys.2024.111878"),
    ("Xiao_2021_LabelEmbeddingCorrelation", "10.1016/j.neucom.2021.07.031"),
    ("Wang_2021_CorrelationGuided", "10.24963/ijcai.2021/463"),
    ("Zhang_2024_BiAttentionCapsule", "10.1016/j.neucom.2024.127671"),
    ("Wang_2021_HybridEmbeddingHTC", "10.1016/j.eswa.2021.115905"),
    ("Read_2011_ClassifierChains", "10.1007/s10994-011-5256-5"),
    ("Zhang_2014_MLLReview", "10.1109/TKDE.2013.39"),
    ("Silla_2011_HTCSurvey", "10.1007/s10618-010-0175-9"),
    ("Zhang_2007_MLKNN", "10.1016/j.patcog.2006.12.019"),
    ("Boutell_2004_Scene", "10.1016/j.patcog.2004.03.009"),
    ("Tsoumakas_2011_RAkEL", "10.1109/TKDE.2010.164"),
    ("Krempl_2015_DelayedLabels", "10.1109/ICMLA.2015.174"),
    ("Li_2022_DelayedLabels", "10.1109/ICNSC55942.2022.10004167"),
    ("Collins_2015_TRIPOD", "10.7326/M14-0697"),
    ("Kapoor_2024_DataLeakage", "10.1038/s41467-024-46150-w"),
    ("Reason_2000_HumanError", "10.1136/bmj.320.7237.768"),
    ("Rasmussen_1997_RiskManagement", "10.1016/S0925-7535(97)00052-0"),
    ("Leveson_2004_STAMP", "10.1016/S0925-7535(03)00047-X"),
    ("Shappell_2007_HFACS", "10.1518/001872007X312469"),
    ("Salmon_2020_HFACSReview", "10.1177/1071181319631086"),
    ("Underwood_2018_STAMPHFACS", "10.1016/j.ssci.2018.04.015"),
    ("Bruce_2024_AccidentGenre", "10.1177/17504813231208563"),
    ("Li_2014_HumanFactors", "10.1016/j.cja.2014.02.002"),
    ("Schroeder_2014_EmergingHF", "10.1177/1541931214581026"),
    ("Zhao_2025_NTSBNarratives", "10.1016/j.aets.2025.12.006"),
    ("Kierszbaum_2023_BERTAviation", "10.2514/6.2023-3438"),
    ("Ye_2024_MatchXML", "10.1109/TKDE.2024.3374750"),
    ("Zhao_2024_VCLDL", "10.1109/TKDE.2023.3323401"),
    ("Zhang_2024_HALB", "10.1016/j.knosys.2024.112153"),
    ("Wang_2025_LSPCL", "10.1016/j.knosys.2024.112887"),
    ("Ren_2025_DyLas", "10.1016/j.inffus.2025.103081"),
    ("Liu_2025_SIHTC", "10.1109/TKDE.2025.3579810"),
]


def ascii_clean(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip()


def bib_escape(text: str) -> str:
    text = ascii_clean(text)
    return text.replace("&", r"\&")


def fetch_csl(doi: str) -> dict:
    req = urllib.request.Request(
        f"https://doi.org/{doi}",
        headers={"Accept": "application/vnd.citationstyles.csl+json", "User-Agent": "IS-paper-reference-builder"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def issued_year(item: dict) -> str:
    parts = item.get("issued", {}).get("date-parts", [[None]])[0]
    return str(parts[0] or "n.d.")


def author_field(item: dict) -> str:
    authors = item.get("author", [])
    names = []
    for author in authors:
        family = bib_escape(author.get("family", ""))
        given = bib_escape(author.get("given", ""))
        if family and given:
            names.append(f"{family}, {given}")
        elif family:
            names.append(family)
    return " and ".join(names) if names else "{Unknown}"


def bib_entry(key: str, doi: str, item: dict) -> str:
    title = bib_escape(item.get("title", [""])[0] if isinstance(item.get("title"), list) else item.get("title", ""))
    container = item.get("container-title", [""])
    venue = bib_escape(container[0] if isinstance(container, list) and container else str(container))
    year = issued_year(item)
    pages = bib_escape(str(item.get("page", "")))
    volume = bib_escape(str(item.get("volume", "")))
    number = bib_escape(str(item.get("issue", "")))
    entry_type = "article" if venue else "inproceedings"
    lines = [f"@{entry_type}{{{key},", f"  title = {{{title}}},", f"  author = {{{author_field(item)}}},", f"  year = {{{year}}},"]
    if venue:
        field = "journal" if entry_type == "article" else "booktitle"
        lines.append(f"  {field} = {{{venue}}},")
    if volume:
        lines.append(f"  volume = {{{volume}}},")
    if number:
        lines.append(f"  number = {{{number}}},")
    if pages:
        lines.append(f"  pages = {{{pages}}},")
    lines.append(f"  doi = {{{doi}}},")
    lines.append(f"  url = {{https://doi.org/{doi}}}")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    manual = [
        """@misc{NTSB_CAROL,
  author = {{National Transportation Safety Board}},
  title = {{Case Analysis and Reporting Online (CAROL)}},
  year = {2026},
  url = {https://data.ntsb.gov/carol-main-public/basic-search},
  note = {Accessed May 22, 2026}
}""",
        """@misc{NTSB_AviationData,
  author = {{National Transportation Safety Board}},
  title = {{Aviation Accident Database and Synopses}},
  year = {2026},
  url = {https://www.ntsb.gov/Pages/AviationQuery.aspx},
  note = {Accessed May 22, 2026}
}""",
    ]
    entries = manual[:]
    for key, doi in DOI_REFS:
        entries.append(bib_entry(key, doi, fetch_csl(doi)))
    OUT.write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} references to {OUT}")


if __name__ == "__main__":
    main()
