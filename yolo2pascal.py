import argparse
from pathlib import Path
from tqdm import tqdm
from formats import PascalVOCAnnotation
from imutils import paths


def convert_annotation(yolo_file, classes_file, output_path, image_file, overlay=None):
    p = PascalVOCAnnotation()
    p.from_yolo(yolo_file, classes_file, image_file)
    with open(output_path, 'w') as pascal_file:
        pascal_file.write(str(p))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts YoloV4 annotation files to PascalVOC Format")
    parser.add_argument('--yolo', help="Path of the YoloV4 file or folder containing YoloV4 files"
                                         " to be converted", required=True)
    parser.add_argument('-c', '--classes', help="Path of the classes file to be used as an index.", required=True)
    parser.add_argument('-i', '--images', help="Path of the annotated images. "
                                               "Default is the folder which contains the annotations.")
    parser.add_argument('-o', '--output', help="Path where the PascalVOC file will be saved. If not specified, "
                                               "it will be saved alongside the classes file, "
                                               "with the same name as the original Yolo file.")
    #parser.add_argument('-y', '--overlay', help="Overlay the annotations over this file and "
    #                                        "save a bounding-box-overlaid image with the original one.")

    args = parser.parse_args()

    args = parser.parse_args()
    source = Path(args.yolo)
    files = []
    if source.is_dir():
        files = [x for x in source.glob("*.txt")]
    else:
        files.append(str(source))
    class_path = Path(args.classes)
    if not args.images:
        args.images = str(class_path.parent)

    if not args.output:
        args.output = class_path.parent

    image_path = Path(args.images)
    output_path = Path(args.output)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    available_images = list(paths.list_images(args.images))
    for p in tqdm(files):
        file_name = Path(p).stem
        output_file = output_path.joinpath(f"{file_name}.xml")
        image_files = [x for x in available_images if Path(x).stem == file_name]
        image_file = image_files and image_files[0] or None
        if image_file:
            convert_annotation(str(p), args.classes, str(output_file), image_file)
        else:
            print(f"No image corresponding to {p}, cannot convert YOLO annnotation without reference image.")
