import dicom
import glob
import sys
import csv
import argparse
import os

columns = {
        "Study Date":           dicom.tag.BaseTag(0x00080020),
        "Series Date":          dicom.tag.BaseTag(0x00080021),
        "Acquisition Date":     dicom.tag.BaseTag(0x00080022),
        "Patient ID":           dicom.tag.BaseTag(0x00100020),
        "Sex":                  dicom.tag.BaseTag(0x00100040),
        "Weight":               dicom.tag.BaseTag(0x00101030),
        "Receive Coil Name":    dicom.tag.BaseTag(0x00181250),
        "Rows":                 dicom.tag.BaseTag(0x00280010),
        "Columns":              dicom.tag.BaseTag(0x00280011),
        "Study Instance UID":   dicom.tag.BaseTag(0x0020000d),
        "Series Instance UID":  dicom.tag.BaseTag(0x0020000e),
        "Laterality":           dicom.tag.BaseTag(0x00200060),
        "Pixel Spacing":        dicom.tag.BaseTag(0x00280030),
    }

def load_img_metadata(img):
    inst = {}
    for k, v in columns.items():
        try:
                inst[k] = img[v].value
        except:
                inst[k] = "MISSING"
                #print("Failed to find key {}".format(k))

    return inst

def load_images(dir_path):
    ret = []
    for f_path in glob.glob(dir_path + 'ISPY*/*/*/*.dcm'):
        dcm_img = dicom.read_file(f_path)
        inst = load_img_metadata(dcm_img)
        #inst["img_path"] =  #f_path
        inst["img_path"] = os.path.abspath(f_path)

        ret.append(inst)

    return ret

def preprocess_images(dir_path, metadata_file_path):
    image_instances = load_images(dir_path)

    with open(metadata_file_path) as csvfile:
        reader = csv.reader(csvfile, delimiter = ',', quotechar = '"')
        key_lst = reader.next()
        # Iterate over rows of csv.
        for i, row in enumerate(reader):
            patient_id = row[0]
            # Find the corresponding image map.
            inst_found = None
            for inst in image_instances:
                inst_patient_id = inst["Patient ID"].split("_")[1]
                if inst_patient_id == patient_id:
                    inst_found = inst
                    for i_r, v_r in enumerate(row):
                        inst_found[key_lst[i_r]] = v_r
                        #print(inst_found)


            #assert inst_found is not None
    return image_instances

def output_to_file(img_metadata, output_file):
    f = open(output_file, 'w')
    csv_headers = ['"' + str(v) + '"' for v in img_metadata[0].keys()]
    csv_headers.append('"png_path"')
    f.write(",".join(csv_headers) + "\n")
    for img in img_metadata:
        for png_path in glob.glob(img['img_path'].split('.dcm')[0] + '*.png'):
            vals = ['"' + str(v) + '"' for v in img.values()]
            vals.append('"' + png_path + '"')
            f.write(','.join(vals) + "\n")


def parse_command_line():
    parser = argparse.ArgumentParser(description='Preprocesses data.')
    parser.add_argument('img_dir', metavar = 'img_dir', type = str,
                        help='the path to the image DOI folder.')
    parser.add_argument('csv_path', metavar = 'csv_path', type = str,
                        help='the path to the csv with the rest of the data.')
    parser.add_argument('output_file', metavar = 'output_file', type = str,
                        help='the path to the output file.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_command_line()
    print (args)
    img_dir = args.img_dir
    csv_path = args.csv_path
    output_file = args.output_file
    img_metadata = preprocess_images(img_dir, csv_path)

    print(img_metadata)

    output_to_file(img_metadata, output_file)

