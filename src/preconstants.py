from pathlib import Path
INB_DATA_DIR = Path("")
INB_MASTER_CSV = ''
INB_TRAIN_DIR = INB_DATA_DIR / 'train'
INB_TEST_DIR = INB_DATA_DIR / 'test'
INB_VALID_DIR = INB_DATA_DIR / 'valid'
# INB_5x200 = INB_DATA_DIR / "inb_5x200.csv"

INB_VALID_NUM = 10
INB_PATH_COL = "File_path"
INB_SPLIT_COL = "split"

INB_REPORT_COL = "conclusie"

INB_COMPETITION_TASKS = [
    "pathology",
    "pathology2",
]

INB_CLASS_PROMPTS = {
    "pathology":{
        "location": [
            "central",
            "right",
            "left",
            "axillary",
        ],
        "describ": [
            "abnormal",
            "suspected",
            "asymmetry"
           ],
        "subtype": [
            "mass",
            "malignancy",
            "leision",
            "carcinoma",
            ],
        "severity": [
            "birads4",
            "birads5",
            "birads6",
            "malignant"],
    },

}
