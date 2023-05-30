import xml.etree.ElementTree as et


class labels_xml:
    def __init__(self, path):
        self.tree = et.parse(path)
        self.root = self.tree.getroot()
        self.count_del = 0

    def set_file_name(self, file_name, shape):
        self.root[1].text = file_name
        self.root[2].text = f'/my-project-name/{file_name}'
        self.root[4][0].text = str(shape[1])
        self.root[4][1].text = str(shape[0])
        # print(self.root[4][0].text)
        # print(self.root[4][1].text)

    def label(self, name_labels, box_labels):
        labels_to_xml = box_labels
        for i, item in enumerate(self.root):
            try:
                # print(i)
                name = item[0].text
                # print(name)
                if name == name_labels and len(labels_to_xml) > 0:
                    # xmin, ymin, xmax, ymax
                    # print(f'đã vào {name_labels}')
                    item[4][0].text = str(labels_to_xml[0])
                    item[4][1].text = str(labels_to_xml[1])
                    item[4][2].text = str(labels_to_xml[2])
                    item[4][3].text = str(labels_to_xml[3])
                # print('item',item)
                # self.root[i] = item
                # print(self.root[i])
            except:
                pass

    def to_xml(self):
        i = 4

        count = len(self.root) - 1
        # print(self.root[5][4][1].text)
        # print('========')
        while True:
            try:
                if self.root[i][4][1].text == 'None':
                    self.root.remove(self.root[i])
                    self.count_del += 1
                    i = 4
                    count = len(self.root)
                i += 1
                # print(i)
            except:
                i += 1
                if i > count:
                    break

    def write_file(self, name_file_output):
        self.tree.write(name_file_output)


def label_to_xml(path_form, boxes, name_input, name_output, shape):
    labels_auto = labels_xml(path_form)
    labels_auto.set_file_name(name_input, shape)
    labels_auto.label('Face', boxes[0])
    labels_auto.label('Phone', boxes[1])
    labels_auto.label('Drinks', boxes[2])
    labels_auto.label('Seatbelt', boxes[3])
    labels_auto.to_xml()
    labels_auto.write_file(name_output)


if __name__ == '__main__':
    box = [[1, 2, 3, 4], [], [5, 6, 7, 8], []]
    label_to_xml('form_detect.xml', box, 'image_test.jpg', 'image_test.xml')
    # labels_auto = labels_xml('form_detect.xml')
    # labels_auto.set_file_name('image_test.jpg')
    # labels_auto.label('Face', box[0])
    # labels_auto.label('Phone', box[1])
    # labels_auto.label('Drinks', box[2])
    # labels_auto.label('Seatbelt', box[3])
    # labels_auto.to_xml()
    # labels_auto.write_file('image_test.xml')
