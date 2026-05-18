"""MIRROR biomarkers and zero-shot prompts."""

TITLE = "MIRROR: Zero-shot prediction of spatial immune profiles from routine pathology slides using a vision–language model"

BIOMARKERS = [
    "CD31",
    "CD45",
    "CD68",
    "CD4",
    "FOXP3",
    "CD8a",
    "CD45RO",
    "CD20",
    "PD-L1",
    "CD3e",
    "CD163",
    "PD-1",
    "Ki67",
    "Pan-CK",
    "SMA",
]

TEMPLATES = [
    "CLASSNAME.",
    "a photomicrograph showing CLASSNAME.",
    "a photomicrograph of CLASSNAME.",
    "an image of CLASSNAME.",
    "an image showing CLASSNAME.",
    "an example of CLASSNAME.",
    "CLASSNAME is shown.",
    "this is CLASSNAME.",
    "there is CLASSNAME.",
    "a histopathological image showing CLASSNAME.",
    "a histopathological image of CLASSNAME.",
    "a histopathological photograph of CLASSNAME.",
    "a histopathological photograph showing CLASSNAME.",
    "shows CLASSNAME.",
    "presence of CLASSNAME.",
    "CLASSNAME is present.",
    "an H&E stained image of CLASSNAME.",
    "an H&E stained image showing CLASSNAME.",
    "an H&E image showing CLASSNAME.",
    "an H&E image of CLASSNAME.",
    "CLASSNAME, H&E stain.",
    "CLASSNAME, H&E.",
]
