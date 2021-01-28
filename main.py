from data_process import *
from trainining import *


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

    def run_lenet5(self):
        data_set_dir = self.__data_dir__ + "pfile4LeNet/"
        data_dir = self.__data_dir__ + "results/LeNet4/"
        lenet5 = LeNet5.LeNet5(data_set_dir, data_dir, 43)
        lenet5.run()

    def run_pretraining_grouping(self):
        g = data_process_utils.DataGrouping()
        source_dir = self.__data_dir__ + "renumbered_data/"
        des_dir = self.__data_dir__ + "pretraining_data/"
        g.pretraining(source_dir, des_dir)

    def run_dump2p4pretraining(self):
        input_dir = self.__data_dir__ + "pretraining_data/"
        output_dir = self.__data_dir__ + "pfile4pretraining/"
        data_process_utils.image_dump(input_dir, output_dir)

    def run_pretraining(self):
        data_set_dir = self.__data_dir__ + "pfile4pretraining/"
        data_dir = self.__data_dir__ + "results/pretraining/"
        lenet5 = LeNet5.LeNet5(data_set_dir, data_dir, 42)
        lenet5.run()

    def run_transfer_learning_grouping(self):
        g = data_process_utils.DataGrouping()

        input_dir = self.__data_dir__ + "augmented/"
        output_dir = self.__data_dir__ + "transferlearning/"
        test_input = self.__data_dir__ + "renumbered_data/test/"
        g.transfer_learning(input_dir, output_dir, test_input)

    def run_dump2p4transfer(self):
        input_dir = self.__data_dir__ + "transferlearning/"
        output_dir = self.__data_dir__ + "pfile4transfer/"
        data_process_utils.image_dump(input_dir, output_dir)


# Press the green button in the gutter to run the script.

w = Workbench()
w.run_dump2p4transfer()
