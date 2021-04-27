import argparse

from tqdm import tqdm

from formats import PascalVOCAnnotation
from pathlib import Path


def convert_annotation(voc_file, classes_file, ouput_path, overlay=None):
    p = PascalVOCAnnotation()
    p.parse(voc_file)
    class_path = Path(classes_file)
    if not class_path.is_file():
        try:
            class_path.touch()
        except FileNotFoundError:
            class_path.parent.mkdir(parents=True)
            class_path.touch()
    annotation, classes = p.to_yolo(classes_file)
    with open(classes_file, 'w') as class_file:
        class_file.write("\n".join(classes))
    if not ouput_path:
        ouput_path = class_path.parent.joinpath(f"{Path(voc_file).stem}.txt")
    with open(ouput_path, 'w') as yolo_file:
        yolo_file.write(str(annotation))
    if overlay:
        original_path = Path(overlay)
        overlay_file = args.output.parent.joinpath(f"{original_path.stem}_overlay.{original_path.suffix}")
        p.overlay(str(original_path), str(overlay_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts PascalVOC annotation files to YoloV4 Format")
    parser.add_argument('--pascal', help="Path of the PascalVOC file or folder containing PascalVOC files"
                                         " to be converted", required=True)
    parser.add_argument('-c', '--classes', help="Path of the classes file to be used as an index. "
                                                "If the file does not exist it will be created. "
                                                "It will be updated with categories present in the "
                                                "PascalVOC annotation file", required=True)
    parser.add_argument('-o', '--output', help="Path where the yolo file will be saved. If not specified, "
                                               "it will be saved alongside the classes file, "
                                               "with the same name as the original PascalVOC file.")
    parser.add_argument('-y', '--overlay', help="Overlay the annotations over this file and "
                                                "save a bounding-box-overlaid image with the original one.")

    args = parser.parse_args()
    source = Path(args.pascal)
    files = []
    if source.is_dir():
        files = [x for x in source.glob("*.xml")]
    else:
        files.append(str(source))

    class_path = Path(args.classes)
    for p in tqdm(files):
        output_path = class_path.parent.joinpath(f"{Path(p).stem}.txt")
        convert_annotation(str(p), args.classes, str(output_path))
