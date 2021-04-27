from lxml import objectify, etree
from pathlib import Path
import cv2

class PascalVOCAnnotation(object):
    class PascalVOCObject(object):
        """
        This class represents each of the annotated objects within a PascalVOC annotation file
        """
        def __init__(self):
            self.name = ""
            self.pose = "Unspecified"
            self.truncated = 0
            self.difficult = 0
            self.xmin = 0
            self.ymin = 0
            self.xmax = 0
            self.ymax = 0

        @property
        def bndbox(self):
            return [self.xmin, self.ymin, self.xmax, self.ymax]
        @bndbox.setter
        def bndbox(self, box):
            assert len(box) == 4
            self.xmin, self.ymin, self.xmax, self.ymax = box

        def to_xml(self):
            E = objectify.ElementMaker(annotate=False)
            return E.object(
                E.name(self.name),
                E.pose(self.pose),
                E.truncated(self.truncated),
                E.difficult(self.difficult),
                E.bndbox(
                    E.xmin(self.xmin),
                    E.ymin(self.ymin),
                    E.xmax(self.xmax),
                    E.ymax(self.ymax)
                )
            )

        def parse(self, element):
            self.name = element.name.text
            self.pose = element.pose.text
            self.truncated = element.truncated.text
            self.difficult = element.difficult.text
            self.xmin = int(element.bndbox.xmin.text)
            self.ymin = int(element.bndbox.ymin.text)
            self.xmax = int(element.bndbox.xmax.text)
            self.ymax = int(element.bndbox.ymax.text)
            return self

        def from_yolo(self, object, classes, size):
            index, xcenter, ycenter, width, height = object
            img_width, img_height, _ = size
            self.name = classes[index]
            self.xmin = int((xcenter - width / 2) *   img_width)
            self.ymin = int((ycenter - height / 2) * img_height)
            self.xmax = int((xcenter +  width / 2) *  img_width)
            self.ymax = int((ycenter +  height / 2) *img_height)

    """
    Class defining a PascalVOC annotation file.
    PascalVOC is an XML file with the following structure:
        <annotation>
            <folder></folder> --> The folder containing the annotated image
            <filename></filename> --> the relative path of the annotated image
            <path></path> --> an absolute path of the output file after annotations
            <size>
                <width></width> 
                <height></height>
                <depth></depth> --> 3 is RGB, 1 is B&W
            </size> --> shape of the image in (pixels, pixels, number of channels) 
            <segmented></segmented> --> TODO define
            <object> --> a list(0..N) of annotations inside the image
                <name></name> --> the name of the category for the object being annotated
                <pose></pose> --> TODO define
                <truncated></truncated> --> 1 if the object extends beyond the truncated image, else 0
                <difficult></difficult> --> TODO define
                <bndbox>
                    <xmin></xmin>
                    <ymin></ymin>
                    <xmax></xmax>
                    <ymax></ymax>
                </bndbox> --> bounding box of the annotated object in pixels
            </object>
        </annotation>
    """
    def __init__(self):
        self.folder = None
        self.filename = None
        self.path = None
        self.width = 0
        self.height = 0
        self.depth = 1
        self.segmented = 0
        self.objects = []

    @property
    def size(self):
        return [self.width, self.height, self.depth]

    @size.setter
    def size(self, size):
        assert len(size) == 3
        self.width, self.height, self.depth = size

    def __repr__(self):
        E = objectify.ElementMaker(annotate=False)
        as_xml = E.annotation(
            E.folder(self.folder),
            E.filename(self.filename),
            E.size(
                E.width(self.width),
                E.height(self.height),
                E.depth(self.depth)
            ),
            E.segmented(self.segmented),
        )
        for e in self.objects:
            as_xml.append(e.to_xml())
        return etree.tostring(as_xml, pretty_print=True, encoding=str)

    def parse(self, path):
        obj = objectify.parse(path).getroot()
        self.folder = obj.folder.text
        self.filename = obj.filename.text
        self.path = obj.path.text
        self.width = int(obj.size.width.text)
        self.height = int(obj.size.height.text)
        self.depth = int(obj.size.depth.text)
        self.segmented = obj.segmented.text
        try:
            for o in obj.object:
                self.objects.append(PascalVOCAnnotation.PascalVOCObject().parse(o))
        except AttributeError:
            pass ## The annotation has no objects

    def to_yolo(self, classes):
        with open(classes, 'r') as class_file:
            classes = class_file.read().split()
            yolo_object = YoloAnnotation()
            for o in self.objects:
                try:
                    class_index = classes.index(o.name)
                except ValueError:
                    classes.append(o.name)
                    class_index = len(classes) - 1
                xcenter = (o.xmax + o.xmin) / 2
                xcenter /= self.width
                ycenter = (o.ymax + o.ymin) / 2
                ycenter /= self.height
                width = (o.xmax - o.xmin) / self.width
                height = (o.ymax - o.ymin) / self.height
                yolo_object.append(class_index, xcenter, ycenter, width, height)
            return yolo_object, classes

    def _from_yolo_object(self, yolo_object, classes_file, image_file):
        class_path = Path(classes_file)
        img_path = Path(image_file)
        assert class_path.is_file() and img_path.is_file()
        with open(classes_file, 'r') as file:
            classes = file.read().split()
            im = cv2.imread(image_file)
            self.height, self.width, self.depth = im.shape
            self.objects = []
            for o in yolo_object:
                annotated_obj = PascalVOCAnnotation.PascalVOCObject()
                annotated_obj.from_yolo(o, classes, self.size)
                self.objects.append(annotated_obj)

    def from_yolo(self, yolo_file, classes_file, image_file):
        y = YoloAnnotation()
        y.parse(yolo_file)
        self._from_yolo_object(y, classes_file, image_file)

    def overlay(self, image_src, image_dst=None):
        print(image_src)
        im = cv2.imread(image_src)
        result = im.copy()
        for o in self.objects:
            cv2.rectangle(result, (o.xmin, o.ymin), (o.xmax, o.ymax), (0, 0, 255), 2)
        if image_dst:
            cv2.imwrite(image_dst, result)
        else:
            cv2.imshow("", result)
            cv2.waitKey(0)

class YoloAnnotation(object):
    """
    This class represents an annotation file in the Yolo DarkNET format. It contains a list of annotated objects with:
    - category name corresponding to an index in a classes file
    - xcenter normalized to the width of the image
    - ycenter normalized to the height of the image
    - width width of the annotated bounding box normalized to the width of the image
    - height height of the annotated bounding box normalized to the height of the image
    """
    class YoloObject(object):
        """
        This class represents each of the annotated objects in a Yolo annotation file
        """
        def __init__(self):
            self.class_ = -1
            self.xcenter = 0
            self.ycenter = 0
            self.width = 0
            self.height = 0

        def __repr__(self):
            return f"{self.class_} {self.xcenter} {self.ycenter} {self.ycenter} {self.width} {self.height}"

        def parse(self, annotation):
            self.class_, self.xcenter, self.ycenter, self.width, self.height = [float(x) for x in annotation.split(" ")]
            self.class_ = int(self.class_)
            return self

    def __init__(self):
        self.width = 0
        self.height = 0
        self.classes_file = ""
        self.objects = []
        self.current = 0

    def __repr__(self):
        return "\n".join([str(x) for x in self.objects])

    def parse(self, annotation_path):
        with open(annotation_path, "r") as file:
            self.objects = [YoloAnnotation.YoloObject().parse(ann) for ann in file.readlines()]

    def append(self, class_index, xcenter, ycenter, width, height):
        obj = YoloAnnotation.YoloObject()
        obj.class_ = class_index
        obj.xcenter = xcenter
        obj.ycenter = ycenter
        obj.width = width
        obj.height = height
        self.objects.append(obj)

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        if self.current < len(self.objects):
            o = self.objects[self.current]
            self.current += 1
            return [o.class_, o.xcenter, o.ycenter, o.width, o.height]
        raise StopIteration


if __name__ == '__main__':
    pickled = r"C:\Users\afaciov\Pictures\dataset\container ship\Image_1.xml"
    pickled_yolo = r"C:\Users\afaciov\Pictures\dataset\cruceros\img_03_1100.txt"
    image_for_pickled_yolo = r"C:\Users\afaciov\Pictures\dataset\cruceros\img_03_1100.jpg"
    classes = r"C:\Users\afaciov\Pictures\dataset\cruceros\classes.txt"
    p = PascalVOCAnnotation()
    #p.parse(pickled)
    #as_yolo, classes = p.to_yolo()
    #print(str(as_yolo))
    #print(classes)
    y = YoloAnnotation()
    y.parse(pickled_yolo)
    print(y)
    p.from_yolo(y, classes, image_for_pickled_yolo)
    print(p)
    p.overlay(image_for_pickled_yolo)
    #my_yolo, classes = p.to_yolo(classes)
    #my_obj = PascalVOC.PascalVOCObject()
    #my_obj.name = "MyCategory"
    #my_obj.pose = "MyPose"
    #my_obj.truncated = 1
    #my_obj.difficult = 1
    #my_obj.bndbox = [300, 400, 600, 700]
    #p.folder = "MyFolder"
    #p.filename ="MyFilename"
    #p.path = "MyPath"
    #p.size = (1000, 600, 3)
    #p.segmented = 1
    #p.objects.append(my_obj)

    #print(y)

