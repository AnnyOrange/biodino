#!/usr/bin/env python3
"""Export the current default bio benchmark dataset inventory to Excel."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/00_reports/test_dataset_inventory_20260824.xlsx"

MEDMNIST = {
    "bloodmnist": ("BloodMNIST", "外周血涂片显微图像", "8类血细胞分类"),
    "pathmnist": ("PathMNIST", "结直肠癌H&E组织病理图像", "9类组织分类"),
    "tissuemnist": ("TissueMNIST", "人肾皮质显微图像", "8类肾组织分类"),
    "breastmnist": ("BreastMNIST", "乳腺超声图像", "良恶性二分类"),
    "organamnist": ("OrganAMNIST", "腹部CT轴位切片", "11类器官分类"),
    "organcmnist": ("OrganCMNIST", "腹部CT冠状位切片", "11类器官分类"),
    "organsmnist": ("OrganSMNIST", "腹部CT矢状位切片", "11类器官分类"),
    "dermamnist": ("DermaMNIST", "皮肤镜图像", "7类皮肤病变分类"),
    "octmnist": ("OCTMNIST", "视网膜OCT图像", "4类视网膜状态分类"),
    "pneumoniamnist": ("PneumoniaMNIST", "儿童胸部X光", "肺炎二分类"),
    "retinamnist": ("RetinaMNIST", "眼底图像", "5级糖尿病视网膜病变分类"),
    "chestmnist": ("ChestMNIST", "胸部X光", "14种胸部病征多标签分类"),
}

SOURCES = {
    "MedMNIST": ("Yang et al., MedMNIST v2", 2023, "https://doi.org/10.1038/s41597-022-01721-8"),
    "BBBC": ("Ljosa et al., Annotated high-throughput microscopy image sets for validation", 2012, "https://doi.org/10.1038/nmeth.2083"),
    "CYCLoPs": ("Chong et al., The yeast proteome in a single cell", 2015, "https://doi.org/10.1016/j.cell.2015.08.049"),
    "MIDOG": ("MIDOG 2025 atypical mitosis classification challenge dataset", 2025, "https://midog2025.grand-challenge.org/"),
    "PCam": ("Veeling et al., Rotation Equivariant CNNs for Digital Pathology", 2018, "https://arxiv.org/abs/1806.03962"),
    "NCT": ("Kather et al., 100,000 histological images of colorectal cancer and healthy tissue", 2018, "https://doi.org/10.5281/zenodo.1214456"),
    "LC25000": ("Borkowski et al., Lung and Colon Cancer Histopathological Image Dataset", 2019, "https://arxiv.org/abs/1912.12142"),
    "CHAMMI": ("Chen and Pham et al., CHAMMI: A benchmark for channel-adaptive models", 2023, "https://doi.org/10.5281/zenodo.7988357"),
    "CoNIC": ("Graham et al., CoNIC Challenge", 2024, "https://doi.org/10.1016/j.media.2023.103047"),
    "LIVECell": ("Edlund et al., LIVECell", 2021, "https://doi.org/10.1038/s41592-021-01249-6"),
    "HPA": ("Thul et al., A subcellular map of the human proteome", 2017, "https://doi.org/10.1126/science.aal3321"),
    "RxRx1": ("Sypetkowski et al., RxRx1", 2023, "https://arxiv.org/abs/2301.05768"),
    "DSB2018": ("Caicedo et al., Nucleus segmentation across imaging experiments: DSB2018", 2019, "https://doi.org/10.1038/s41592-019-0612-7"),
    "MoNuSeg": ("Kumar et al., A Dataset and a Technique for Generalized Nuclear Segmentation", 2017, "https://doi.org/10.1109/TMI.2017.2677499"),
    "PanNuke": ("Gamper et al., PanNuke", 2019, "https://arxiv.org/abs/2003.10778"),
    "TissueNet": ("Greenwald et al., Whole-cell segmentation of tissue images with human-level performance", 2022, "https://doi.org/10.1038/s41587-021-01094-0"),
    "MultimodalCellSeg": ("Ma et al., Multi-modality Cell Segmentation Challenge", 2023, "https://arxiv.org/abs/2308.05864"),
    "Cellpose": ("Stringer et al., Cellpose", 2021, "https://doi.org/10.1038/s41592-020-01018-x"),
}


def row(task, code, name, family, modality, target, protocol, metric, note=""):
    article, year, url = SOURCES[family]
    return {
        "任务": task,
        "代码名": code,
        "正式名称": name,
        "来源数据族": family,
        "图像模态": modality,
        "预测或评估目标": target,
        "测试协议": protocol,
        "主要指标": metric,
        "来源文章或数据说明": article,
        "年份": year,
        "链接": url,
        "备注": note,
    }


def classification_rows():
    rows = []
    for code, (name, modality, target) in MEDMNIST.items():
        metric = "macro-AUC / macro-F1" if code == "chestmnist" else "balanced accuracy / macro-F1"
        rows.append(row("Classification", code, name, "MedMNIST", modality, target, "官方train -> test", metric))
    rows.extend(
        [
            row("Classification", "bbbc048-cellcycle", "BBBC048 Cell Cycle", "BBBC", "荧光显微图像", "7类细胞周期阶段", "固定source-group held-out split", "balanced accuracy / macro-F1"),
            row("Classification", "cyclops-protein-loc", "CYCLoPs Protein Localization", "CYCLoPs", "酵母荧光显微图像", "蛋白质亚细胞定位分类", "固定source-group held-out split", "balanced accuracy / macro-F1"),
            row("Classification", "midog25-atypical", "MIDOG25 Atypical Mitosis", "MIDOG", "H&E组织病理核图像", "正常有丝分裂NMF vs 非典型有丝分裂AMF", "固定source-group held-out split", "balanced accuracy / macro-F1"),
            row("Classification", "pcam", "PatchCamelyon", "PCam", "淋巴结H&E组织病理patch", "是否含乳腺癌转移灶", "官方train -> test", "balanced accuracy / macro-F1"),
            row("Classification", "nct-crc-he", "NCT-CRC-HE-100K -> CRC-VAL-HE-7K", "NCT", "结直肠H&E组织病理patch", "9类结直肠组织分类", "100K NONORM训练；7K独立患者测试", "balanced accuracy / macro-F1"),
            row("Classification", "lc25000", "LC25000", "LC25000", "肺和结肠H&E组织病理图像", "5类正常/癌症组织分类", "固定分层held-out split", "balanced accuracy / macro-F1"),
        ]
    )
    chammi = [
        ("chammi-allen-task1", "CHAMMI Allen Task 1", "Allen WTC-11 hiPSC多通道显微图像", "细胞周期阶段", "Train -> Task_one"),
        ("chammi-allen-task2", "CHAMMI Allen Task 2", "Allen WTC-11 hiPSC多通道显微图像", "细胞周期阶段", "Train -> Task_two"),
        ("chammi-cp-task1", "CHAMMI Cell Painting Task 1", "Cell Painting 5通道图像", "处理/扰动化合物ID", "Train -> Task_one"),
        ("chammi-cp-task2", "CHAMMI Cell Painting Task 2", "Cell Painting 5通道图像", "处理/扰动化合物ID", "Train -> Task_two"),
        ("chammi-cp-task3", "CHAMMI Cell Painting Task 3", "Cell Painting 5通道图像", "处理/扰动化合物ID", "Train -> Task_three"),
        ("chammi-hpa-task1", "CHAMMI HPA Task 1", "HPA 4通道免疫荧光图像", "蛋白质亚细胞定位", "Train -> Task_one"),
        ("chammi-hpa-task2", "CHAMMI HPA Task 2", "HPA 4通道免疫荧光图像", "蛋白质亚细胞定位", "Train -> Task_two"),
    ]
    for code, name, modality, target, protocol in chammi:
        rows.append(row("Classification", code, name, "CHAMMI", modality, target, protocol, "balanced accuracy / macro-F1", "仅包含closed-set任务"))
    return rows


def regression_rows():
    return [
        row("Regression", "bbbc013", "BBBC013 Dose Regression", "BBBC", "U2OS FKHR-GFP荧光显微图像", "分化合物预测log(1+dose)", "Wortmannin A-D、LY294002 E-H；8-fold replicate-row OOF", "Spearman rho / R2 / MAE"),
        row("Regression", "bbbc005", "BBBC005 Cell Count", "BBBC", "SIMCEP合成荧光显微图像", "连续细胞数", "固定source-group held-out split", "Spearman rho / R2 / MAE"),
        row("Regression", "conic-cell-count", "CoNIC Cell Count", "CoNIC", "结直肠H&E组织病理patch", "中央224px区域的细胞核总数", "source-image grouped 10-fold split", "Spearman rho / R2 / MAE"),
        row("Regression", "livecell-cell-count", "LIVECell Cell Count", "LIVECell", "无标记相差显微图像", "全图细胞实例数", "官方train -> test", "Spearman rho / R2 / MAE"),
    ]


RETRIEVAL_BASE = [
    ("lc25000", "LC25000", "LC25000", "肺和结肠H&E", "5类组织标签", "同集合leave-one-out"),
    ("nct-crc-he-100", "NCT-CRC-HE-100", "NCT", "结直肠H&E patch", "9类组织标签；100张子集", "同集合leave-one-out"),
    ("nct-crc-he-1k", "NCT-CRC-HE-1K", "NCT", "结直肠H&E patch", "9类组织标签；1,000张子集", "同集合leave-one-out"),
    ("crc-val-he-7k", "CRC-VAL-HE-7K", "NCT", "结直肠H&E patch", "9类组织标签；7,180张独立患者验证集", "同集合leave-one-out"),
    ("hpa-subcellular", "HPA Subcellular", "HPA", "4通道免疫荧光细胞图像", "检索同一基因；聚类单一亚细胞定位", "自定义same-gene query/gallery；single-location clustering"),
    ("rxrx1-cross", "RxRx1 Cross-experiment", "RxRx1", "6通道细胞荧光图像", "siRNA扰动ID", "跨实验query/gallery；默认17,728-view balanced core"),
]


def retrieval_clustering_rows():
    rows = []
    for code, name, family, modality, target, protocol in RETRIEVAL_BASE:
        rows.append(row("Retrieval", code, name, family, modality, target, protocol, "R@1/5/10, mAP@1/5/10, MRR"))
        rows.append(row("Clustering", code, name, family, modality, target, protocol, "NMI / ARI / cluster accuracy / silhouette", "与Retrieval共用冻结特征"))
    return rows


def dense_rows():
    rows = [
        row("Detection", "livecell", "LIVECell", "LIVECell", "无标记相差显微图像", "COCO实例框中心 -> ViT patch中心标签", "官方train/val/test；冻结backbone线性patch probe", "patch F1 / precision / recall", "不是COCO bbox mAP"),
        row("Detection", "bbbc038", "BBBC038 / DSB2018", "DSB2018", "多实验、多模态细胞核图像", "实例mask质心 -> ViT patch中心标签", "固定train/val/test；冻结backbone线性patch probe", "patch F1 / precision / recall", "不是COCO bbox mAP"),
        row("Detection", "conic", "CoNIC", "CoNIC", "结直肠H&E组织病理patch", "实例mask质心 -> ViT patch中心标签", "固定train/val/test；冻结backbone线性patch probe", "patch F1 / precision / recall", "不是COCO bbox mAP"),
    ]
    seg = [
        ("bbbc038", "BBBC038 / DSB2018", "DSB2018", "多实验、多模态细胞核图像", "细胞核分割"),
        ("conic", "CoNIC", "CoNIC", "结直肠H&E组织病理patch", "细胞核分割/核类型"),
        ("monuseg", "MoNuSeg", "MoNuSeg", "多器官H&E组织病理图像", "细胞核分割"),
        ("pannuke", "PanNuke", "PanNuke", "多组织H&E组织病理patch", "细胞核分割/5类核类型"),
        ("tissuenet", "TissueNet", "TissueNet", "多组织、多通道荧光图像", "全细胞/细胞核分割"),
        ("livecell", "LIVECell", "LIVECell", "无标记相差显微图像", "活细胞实例分割"),
        ("multimodal_cellseg", "Multimodal CellSeg", "MultimodalCellSeg", "明场、荧光、相差、DIC等多模态图像", "通用细胞分割"),
        ("cellpose", "Cellpose", "Cellpose", "多来源荧光/显微图像", "通用细胞实例分割"),
    ]
    for code, name, family, modality, target in seg:
        rows.append(row("Segmentation", code, name, family, modality, target, "固定train/val/test；dense segmentation evaluator", "mDice", "同名数据可能也用于Detection/Regression"))
    return rows


def style_sheet(ws, widths=None):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    header_fill = PatternFill("solid", fgColor="1F5C5A")
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 28
    if widths:
        for i, width in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = width
    for row_cells in ws.iter_rows(min_row=2):
        for cell in row_cells:
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def append_table(ws, records, columns):
    ws.append(columns)
    for record in records:
        ws.append([record.get(column, "") for column in columns])
        link_cell = ws.cell(ws.max_row, columns.index("链接") + 1) if "链接" in columns else None
        if link_cell and link_cell.value:
            link_cell.hyperlink = link_cell.value
            link_cell.style = "Hyperlink"


def main():
    records = classification_rows() + regression_rows() + retrieval_clustering_rows() + dense_rows()
    expected = {"Classification": 25, "Regression": 4, "Retrieval": 6, "Clustering": 6, "Detection": 3, "Segmentation": 8}
    observed = Counter(record["任务"] for record in records)
    if dict(observed) != expected:
        raise RuntimeError(f"Inventory mismatch: {dict(observed)} != {expected}")

    workbook = Workbook()
    ws = workbook.active
    ws.title = "Tests_by_task"
    columns = ["序号", "任务", "代码名", "正式名称", "来源数据族", "图像模态", "预测或评估目标", "测试协议", "主要指标", "来源文章或数据说明", "年份", "链接", "备注"]
    numbered = [{"序号": i, **record} for i, record in enumerate(records, 1)]
    append_table(ws, numbered, columns)
    style_sheet(ws, [7, 16, 24, 28, 18, 30, 34, 42, 34, 48, 10, 42, 28])

    summary = workbook.create_sheet("Task_summary")
    summary.append(["任务", "测试项数量", "说明"])
    explanations = {
        "Classification": "25个分类数据集/官方任务；ChestMNIST为多标签",
        "Regression": "4个连续值预测任务",
        "Retrieval": "6个图像检索数据集",
        "Clustering": "与Retrieval相同的6个数据集，但采用无监督聚类协议",
        "Detection": "3个center-to-patch线性检测任务，不是bbox mAP",
        "Segmentation": "8个dense segmentation数据集",
    }
    for task in expected:
        summary.append([task, observed[task], explanations[task]])
    summary.append(["总计", len(records), "任务 × 数据集测试项总数"])
    summary.append(["唯一代码名", len({record["代码名"] for record in records}), "跨任务去重后为42个代码数据集/子集"])
    style_sheet(summary, [20, 16, 72])

    unique = workbook.create_sheet("Unique_code_datasets")
    unique_columns = ["代码名", "正式名称", "出现任务", "来源数据族", "图像模态", "链接"]
    by_code = {}
    for record in records:
        code = record["代码名"]
        if code not in by_code:
            by_code[code] = {column: record.get(column, "") for column in unique_columns}
            by_code[code]["出现任务"] = []
        by_code[code]["出现任务"].append(record["任务"])
    unique_records = []
    for code, item in sorted(by_code.items()):
        item["出现任务"] = ", ".join(dict.fromkeys(item["出现任务"]))
        unique_records.append(item)
    append_table(unique, unique_records, unique_columns)
    style_sheet(unique, [26, 32, 34, 22, 38, 46])

    notes = workbook.create_sheet("Scope_and_notes")
    notes.append(["项目", "内容"])
    notes.append(["统计口径", "scripts/run_bio_benchmark_all.sh 当前默认完整suite，统计日期2026-08-24"])
    notes.append(["Retrieval/Clustering", "二者使用相同6个数据集，但一个做kNN检索、一个做MiniBatchKMeans，因此在主表分别计数"])
    notes.append(["Detection", "当前实现为实例中心到ViT patch的线性probe，指标为patch F1，不应写成COCO object-detection mAP"])
    notes.append(["CHAMMI", "仅列入7个closed-set任务；HPA Task 3与CP Task 4因test标签不在Train中而排除"])
    notes.append(["NCT子集", "nct-crc-he-100、nct-crc-he-1k、crc-val-he-7k来自同一NCT-CRC数据族，不是3篇不同来源文章"])
    notes.append(["重复数据源", "CoNIC、LIVECell、BBBC038等会在Regression/Detection/Segmentation多任务重复出现；这是有意的任务级统计"])
    style_sheet(notes, [24, 110])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(OUT)
    print(f"Wrote {OUT}")
    print(f"Rows: {len(records)}; unique code names: {len(by_code)}; tasks: {dict(observed)}")


if __name__ == "__main__":
    main()
