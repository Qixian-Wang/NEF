import os
import numpy as np

from mpi4py import MPI

from miv.core.pipeline import Pipeline
from miv.signal.filter import ButterBandpass
from miv.signal.spike import ThresholdCutoff
from miv.io.intan.data import DataIntan
from miv.core.operator.policy import SupportMPIMerge

from moving_average_firing_rate import MovingAverageFiringRate
from firing_rate_analysis import FiringRateAnalysis


def process_data(data, phase_time):
    pipline_save_path = f"results/{phase_time[0]}"
    figure_save_path = f"data_results/"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    data.configure_load(mpi_comm=comm)
    bandpass_filter = ButterBandpass(lowcut=200, highcut=1600, order=4)
    spike_detection = ThresholdCutoff(cutoff=5.0, num_save=3)
    spike_detection.runner = SupportMPIMerge(comm=comm)

    data >> bandpass_filter >> spike_detection
    pipeline = Pipeline(spike_detection)
    pipeline.run(working_directory=pipline_save_path)
    
    spike_train = spike_detection.output()

    excluded_channels = ['A-027', 'A-028', 'A-031', 'B-000', 'B-001', 'B-004', 'B-005', 'B-006', 'B-007', 'B-008', 'B-009', 'B-010', 'B-011', 'B-012', 'B-013', 'B-014', 'B-015', 'B-022', 'B-024', 'C-002', 'C-006', 'C-010', 'C-030', 'D-024', 'A-000', 'B-015', 'B-016', 'B-031', 'C-000', 'C-001', 'C-016', 'C-031']
    
    reading_channels = ['A-000', 'A-001', 'A-002', 'A-003', 'A-004', 'A-005', 'A-006', 'A-007', 
                        'A-008', 'A-009', 'A-010', 'A-011', 'A-012', 'A-013', 'A-014', 'A-015', 
                        'A-016', 'A-017', 'A-018', 'A-019', 'A-020', 'A-021', 'A-022', 'A-023', 
                        'A-024', 'A-025', 'A-026', 'A-027', 'A-028', 'A-029', 'A-030', 'A-031',
                        'D-000', 'D-001', 'D-002', 'D-003', 'D-004', 'D-005', 'D-006', 'D-007', 
                        'D-008', 'D-009', 'D-010', 'D-011', 'D-012', 'D-013', 'D-014', 'D-015', 
                        'D-016', 'D-017', 'D-018', 'D-019', 'D-020', 'D-021', 'D-022', 'D-023', 
                        'D-024', 'D-025', 'D-026', 'D-027', 'D-028', 'D-029', 'D-030', 'D-031']

    step_size = 0.2
    stimulation_interval = 4
    analysis_start_time = 0.11
    analysis_duration = 1
    
    data_process = MovingAverageFiringRate(spike_train, comm, phase_time)
    data_process.select_reading_channels(excluded_channels, reading_channels)
    spike_rates_list = data_process.moving_average_firing_rate(figure_save_path, step_size=step_size, plot_firing_rate=False)

    if rank == 0:
        firing_rate_analysis = FiringRateAnalysis(spike_rates_list, phase_time, step_size=step_size)
        firing_rate_analysis.tsne_analysis(stimulation_interval, analysis_start_time, analysis_duration, figure_save_path)
        firing_rate_analysis.pca_analysis(stimulation_interval, analysis_start_time, analysis_duration, figure_save_path)

if __name__ == "__main__":
    data_paths = [
        "/scratch1/10197/qxwang/recording_data/Experiment_8_24/qixian_250824_195114",
        "/scratch1/10197/qxwang/recording_data/Experiment_8_24/qixian_250824_200514",
        "/scratch1/10197/qxwang/recording_data/Experiment_8_24/qixian_250824_210934",
        "/scratch1/10197/qxwang/recording_data/Experiment_8_24/qixian_250824_214834",
        "/scratch1/10197/qxwang/recording_data/Experiment_8_24/qixian_250824_225254",
        "/scratch1/10197/qxwang/recording_data/Experiment_8_24/qixian_250824_233154",
        "/scratch1/10197/qxwang/recording_data/Experiment_8_24/qixian_250825_003614",
        "/scratch1/10197/qxwang/recording_data/Experiment_8_24/qixian_250825_011514",
        "/scratch1/10197/qxwang/recording_data/Experiment_8_24/qixian_250825_021934",
        "/scratch1/10197/qxwang/recording_data/Experiment_8_24/qixian_250825_025834",
        "/scratch1/10197/qxwang/recording_data/Experiment_8_24/qixian_250825_040254",
    ]
    # data_paths = [
    #     "/scratch1/10197/qxwang/recording_data/Experiment_8_21/qixian_250821_111442",
    #     "/scratch1/10197/qxwang/recording_data/Experiment_8_21/qixian_250821_113057",
    #     "/scratch1/10197/qxwang/recording_data/Experiment_8_21/qixian_250821_123846",
    #     "/scratch1/10197/qxwang/recording_data/Experiment_8_21/qixian_250821_132002",
    #     "/scratch1/10197/qxwang/recording_data/Experiment_8_21/qixian_250821_142751",
    #     "/scratch1/10197/qxwang/recording_data/Experiment_8_21/qixian_250821_150907",
    #     "/scratch1/10197/qxwang/recording_data/Experiment_8_21/qixian_250821_161656",
    #     "/scratch1/10197/qxwang/recording_data/Experiment_8_21/qixian_250821_165812",
    #     "/scratch1/10197/qxwang/recording_data/Experiment_8_21/qixian_250821_180601",
    #     "/scratch1/10197/qxwang/recording_data/Experiment_8_21/qixian_250821_184717",
    #     "/scratch1/10197/qxwang/recording_data/Experiment_8_21/qixian_250821_195506",
    # ]
    # data_paths = /["/Users/aia/Desktop/test_file/qixian_250813_205329"]

    # phase_time = [{"start": 5, "end": 15}, {"start": 15, "end": 51}]
    # phase_time = [{"start": 83, "end": 403}, {"start": 503, "end": 2103}, 
    #               {"start": 2703, "end": 3023}, {"start": 4823, "end": 6423},
    #               {"start": 7023, "end": 7343}, {"start": 9143, "end": 10743},
    #               {"start": 11343, "end": 11663}] 
    
    phase_time = [["train0", 480], ["pretrain1", 3200], ["train1", 480], ["pretrain2", 3200], ["train2", 480], ["pretrain3", 3200], ["train3", 480], ["pretrain4", 3200], ["train4", 480], ["pretrain5", 3200], ["train5", 480]] 
    # phase_time = [["train1", 40], ["train2", 40]] 

    for file_idx, data_path in enumerate(data_paths):
        print(f"Processing: {data_path}")
        data = DataIntan(data_path)
        process_data(data, phase_time[file_idx])

