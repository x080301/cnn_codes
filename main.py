from data_process import *


class Workbench:
    __data_dir__ = "D:/cnn/data/"
    __codes_dir__ = "D:/cnn/cnn_codes/"

    def run_raw_data_grouping(self):
        g = data_process_utils.DataGrouping()
        source_dir = self.__data_dir__ + "raw_data/"
        des_dir = self.__data_dir__ + "renumbered_data/"
        g.raw_data(source_dir, des_dir)

    def run_dump2p4lenet(self):
        input_dir = self.__data_dir__ + "renumbered_data/"
        output_dir = self.__data_dir__ + "pfile4LeNet/"
        data_process_utils.image_dump(input_dir, output_dir)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    w = Workbench()
    w.run_dump2p4lenet()
