import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.append(r'/home/rlfowler/Documents/research/tfo_inverse_modelling')
import argparse
import pandas as pd
import numpy as np
from model_trainer import CombineMethods, HoldOneOut, TorchLossWrapper  # model_trainer is a package now. Not in this repo
import torch.nn as nn   # PyTorch's neural network module
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from model_trainer.DataLoaderGenerators import DataLoaderGenerator
from model_trainer.loss_funcs import LossFunction, SumLoss
from model_trainer.validation_methods import RandomSplit, ValidationMethod
from model_trainer import ModelTrainer 
import matplotlib.pyplot as plt
import torchinfo
from inverse_modelling_tfo.visualization.distributions import generate_model_error_and_prediction
from mdreport import MarkdownReport
from pathlib import Path

# Talk about loss, network, and results. What I've learned. What is the loss?
# Look at loss for each tmp rather than average loss over all tmps.
#

class Parameters:
    # Default parameters
    DATA_PATH = r'/home/rlfowler/Documents/research/tfo_inverse_modelling/Randalls Folder/data/randall_data_intensities.pkl'
    output_labels:str = "all"
    subset_type:str = "filter1"
    apply_log:bool = True
    random_seed:int = 42
    sample_size:float = 0.05
    test_size:float = 0.2
    data_loader_params = {
    'shuffle': True,    # The dataloader will shuffle its outputs at each epoch
    'num_workers': 0,   # The number of workers that the dataloader will use to generate the batches
    'drop_last': True,  # Drop the last batch if it is smaller than the batch size
    }
    batch_size:int = 32
    num_epochs:int = 2
    model_type:str = "Perceptron"   # Perceptron or SplitCNN
    depth_of_layers = [40, 30, 20, 10]
    cnn_out_channels = [4, 8, 16]   # Number of output channels for each convolutional layer
    cnn_split:int = 2   # Number of channels to split the input into, 2 would divide 40 into two 20s
    cnn_kernel_sizes = [10, 5, 3]  # Kernel sizes for each convolutional layer
    # missing cnn padding, strides, dialations?
    dropout = [0] * (len(depth_of_layers) + 1)
    cnn_dropout = [0.5,0.5]
    validation_method:ValidationMethod = None#RandomSplit(0.8)
    validation_type:str = 'holdout0'#None
    criterion:LossFunction = None#TorchLossWrapper(nn.MSELoss(), name="mse")
    loss_function:str = "mse"
    optimizer:str = "SGD"
    lr:float = 5e-4
    momentum:float = 0.9
    weight_decay:float = 0.0
    bin_count:int = 50
    hue:str = 'Fetal Radius'
    report_name = "default_report1"
    report_title = "Default Title"
    # color_offset = 20 # Offset for the color in the hue (https://matplotlib.org/2.0.2/examples/color/named_colors.html)
    # color_jump = 3
    #colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'black', 'yellow', 'lime', 'teal', ]

    def __str__(self):
        return f"Parameters(\n" + \
            f"\toutput_labels: {self.output_labels}\n" + \
            f"\tsubset_type: {self.subset_type}\n" + \
            f"\tapply_log: {self.apply_log}\n" + \
            f"\trandom_seed: {self.random_seed}\n" + \
            f"\tsample_size: {self.sample_size}\n" + \
            f"\ttest_size: {self.test_size}\n" + \
            f"\tbatch_size: {self.batch_size}\n" + \
            f"\tnum_epochs: {self.num_epochs}\n" + \
            f"\tdepth_of_layers: {self.depth_of_layers}\n" + \
            f"\tcnn_out_channels: {self.cnn_out_channels}\n" + \
            f"\tcnn_split: {self.cnn_split}\n" + \
            f"\tcnn_kernel_sizes: {self.cnn_kernel_sizes}\n" + \
            f"\tdropout: {self.dropout}\n" + \
            f"\tcnn_dropout: {self.cnn_dropout}\n" + \
            f"\tvalidation_method: {self.validation_method}\n" + \
            f"\tcriterion: {self.criterion}\n" + \
            f"\toptimizer: {self.optimizer}\n" + \
            f"\tlr: {self.lr}\n" + \
            f"\tmomentum: {self.momentum}\n" + \
            f"\tweight_decay: {self.weight_decay}\n" + \
            f"\tbin_count: {self.bin_count}\n" + \
            f")"
    
    def set_model(self, type:int):
        if type == 1:   # Perceptron model
            self.depth_of_layers = [20, 10, 10]
            self.model_type = "Perceptron"
        elif type == 2: # Perceptron model
            self.depth_of_layers = [20, 30, 20]
            self.model_type = "Perceptron"
        elif type == 3: # SplitCNN model
            self.depth_of_layers = [12, 8]  # Include cnn variables
            self.dropout = [0.5]*len(self.depth_of_layers)  # New default in case set_dropout is not called
            self.model_type = "SplitCNN"
            self.weight_decay = 1e-4
        elif type == 4: # SplitCNN model
            self.depth_of_layers = [24, 16, 12]  # Include cnn variables
            self.dropout = [0.5]*len(self.depth_of_layers)  # New default in case set_dropout is not called
            self.model_type = "SplitCNN"
            self.weight_decay = 1e-4
        elif type == 5: # SplitCNN model
            self.depth_of_layers = [30, 20, 14]  # Include cnn variables
            self.dropout = [0.5]*len(self.depth_of_layers)  # New default in case set_dropout is not called
            self.model_type = "SplitCNN"
            self.weight_decay = 1e-4
        elif type == 6: # SplitCNN model
            self.depth_of_layers = [24, 16, 12]  # Include cnn variables
            self.dropout = [0.5]*len(self.depth_of_layers)  # New default in case set_dropout is not called
            self.model_type = "SplitCNN"
            self.weight_decay = 1e-4
            self.cnn_split = 4
        elif type == 7: # Perceptron Model
            self.depth_of_layers = [40, 30, 20, 10]
            self.model_type = "Perceptron"
        elif type == 8: # Perceptron Model
            self.depth_of_layers = [30, 20, 10, 10, 10]
            self.model_type = "Perceptron"
        else:  
            raise ValueError(f"Depth type {type} not recognized")
        
    def set_dropout(self, type:int):
        L = len(self.depth_of_layers) + 1
        if type == 1:
            self.dropout = [0.5]*L
        elif type == 2:
            self.dropout = [0.4]*L
        elif type == 3:
            self.dropout = [0.3]*L
        elif type == 4:
            self.dropout = [0.1]*L
        elif type == 5:
            self.dropout = [0.01]*L
        elif type == 6:
            self.dropout = [0]*L
        else:
            raise ValueError(f"Dropout type {type} not recognized")

    def set_validation(self, type:str):
        if type == 'random':
            self.validation_method = RandomSplit(0.8)
        elif type[:4] == 'hold':
            self.validation_method = None
            self.validation_type = type
        else:
            raise ValueError(f"Validation type {type} not recognized")
        
    def set_criteria(self, type:str):
        if type == 'mse':
            self.loss_function = "mse"
            self.criterion = TorchLossWrapper(nn.MSELoss(), name="mse")
        elif type == 'mse_seperated':
            self.loss_function = "mse"
            self.criterion = None
        else:
            raise ValueError(f"Criterion type {type} not recognized")

    # def set_hue(self, hue:int):
    #     TMPS = ['Maternal Wall Thickness','Fetal Radius','Fetal Displacement','Maternal Hb Concentration','Maternal Saturation','Fetal Hb Concentration','Fetal Saturation']
    #     self.hue = TMPS[hue]


def run(params:Parameters):
    # Load the data
    data:pd.DataFrame = pd.read_pickle(params.DATA_PATH)

    if params.subset_type[:6] == "filter":
        if params.subset_type[6:] == "1":
            columns = ['Fetal Hb Concentration', 'Fetal Radius', 'Maternal Saturation', 'Maternal Hb Concentration']
            to_keep = [np.sort(np.unique(data['Fetal Hb Concentration']))[1::3],\
                         np.sort(np.unique(data['Fetal Radius']))[:11],\
                         np.sort(np.unique(data['Maternal Saturation']))[::2],\
                         np.sort(np.unique(data['Maternal Hb Concentration']))[::2]]
            for col, keep in zip(columns, to_keep):
                data = data.loc[data[col].isin(keep)]
        elif params.subset_type[6:] == "2": # Changed fetal radius to larger values and reduced fetal saturation
            columns = ['Fetal Hb Concentration', 'Fetal Radius', 'Maternal Saturation', 'Maternal Hb Concentration','Fetal Saturation']
            to_keep = [np.sort(np.unique(data['Fetal Hb Concentration']))[1::3],\
                         np.sort(np.unique(data['Fetal Radius']))[11:],\
                         np.sort(np.unique(data['Maternal Saturation']))[::2],\
                         np.sort(np.unique(data['Maternal Hb Concentration']))[::2],\
                         np.sort(np.unique(data['Fetal Saturation']))[::2]\
                            ]
            for col, keep in zip(columns, to_keep):
                data = data.loc[data[col].isin(keep)]

    # Select data to predict
    x_columns = data.columns[7:]
    if params.output_labels == "all":
        y_columns = data.columns[:7]
        print(f"y_columns: {y_columns.tolist()}")
    # elif type(params.output_labels) == list:
    #     if type(params.output_labels[0]) == int:
    #         y_columns = data.columns[:7][params.output_labels]
    #     elif type(params.output_labels[0]) == str:
    #         y_columns = params.output_labels
    #     else:
    #         raise ValueError("Invalid output labels. Must be either 'all', a list of integers or str, or a single integer or str.")
    elif params.output_labels.isdigit():
        y_columns = [data.columns[:7][int(params.output_labels)]]
        print(f"y_columns: {y_columns}")
    else:
        raise ValueError("Invalid output labels. Must be either 'all', a list of integers or str, or a single integer or str.")
    print(f"x_columns: {x_columns.tolist()}",flush=True)
        
    IN_FEATURES = len(x_columns)
    OUT_FEATURES = len(y_columns)
    print("In Features :", IN_FEATURES)  
    print("Out Features:", OUT_FEATURES)

    # Should change this implementation... Right now, equal weighting across all params
    if params.criterion == None:
        if params.loss_function == "mse":
            criterion_weights = [1]*OUT_FEATURES
            params.criterion = SumLoss([TorchLossWrapper(nn.MSELoss(), [i], name=y_columns[i]) for i in range(OUT_FEATURES)], criterion_weights)

    # Apply log to the data
    if params.apply_log:
        data[x_columns] = np.log(data[x_columns])
    data.dropna(inplace=True)


    ## Scale y, sets mean to 0 and variance to 1
    y_scaler = preprocessing.StandardScaler()
    if OUT_FEATURES == 1:
        data[y_columns] = y_scaler.fit_transform(data[y_columns].to_numpy().reshape(-1, 1))
    else:
        data[y_columns] = y_scaler.fit_transform(data[y_columns])

    ## Scale x
    x_scaler = preprocessing.StandardScaler()
    data[x_columns] = x_scaler.fit_transform(data[x_columns])


    # Set validation method # Should I add randomsplit to all options here?
    if params.validation_method is None:
        if params.validation_type[:7] == 'holdout':
            if len(y_columns) == 1:
                params.validation_method = HoldOneOut(y_columns[0], data[y_columns[0]].unique()[3])
            else:
                try:
                    col = int(params.validation_type[7:])
                    params.validation_method = HoldOneOut(y_columns[col], data[y_columns[col]].unique()[3])
                except:
                    if params.validation_type[7:] == '_all':
                        params.validation_method = CombineMethods([HoldOneOut(y_columns[i], data[y_columns[i]].unique()[3]) for i in range(OUT_FEATURES)])
                    else:
                        raise ValueError(f"Invalid validation type {params.validation_type}")
        elif params.validation_type[:12] == 'holdrndsplit':
            if len(y_columns) == 1:
                params.validation_method = CombineMethods([HoldOneOut(y_columns[0], data[y_columns[0]].unique()[3]), RandomSplit(0.9)])
            else:
                try:
                    col = int(params.validation_type[12:])
                    params.validation_method = CombineMethods([HoldOneOut(y_columns[col], data[y_columns[col]].unique()[3]), RandomSplit(0.9)])
                except:
                    if params.validation_type[12:] == '_all':
                        val_methods = [HoldOneOut(y_columns[i], data[y_columns[i]].unique()[3]) for i in range(OUT_FEATURES)]
                        val_methods.append(RandomSplit(0.9))
                        params.validation_method = CombineMethods(val_methods)
                    else:
                        raise ValueError(f"Invalid validation type {params.validation_type}")
        else:
            raise ValueError(f"Invalid validation type {params.validation_type}")


    # Split the data
    if params.subset_type[:6] != "filter":
        if params.subset_type == "random":
            data = data.sample(frac=params.sample_size, random_state=params.random_seed)
        elif params.subset_type == "all":
            pass
        else:
            raise ValueError("Invalid subset type. Must be either 'random' or 'all'.")
    

    # Create the model
    if params.model_type == "Perceptron":
        from inverse_modelling_tfo.model_training.custom_models import PerceptronBD
        model = PerceptronBD([IN_FEATURES, *params.depth_of_layers, OUT_FEATURES], dropout_rates=params.dropout)
    elif params.model_type == "SplitCNN":
        from inverse_modelling_tfo.model_training.custom_models import SplitChannelCNN
        model = SplitChannelCNN(IN_FEATURES, params.cnn_split, params.cnn_out_channels, params.cnn_kernel_sizes, fc_output_node_counts=[*params.depth_of_layers, OUT_FEATURES], fc_dropouts=params.dropout, cnn_dropouts=params.cnn_dropout)
    else:
        raise ValueError(f"Model type {params.model_type} not recognized")    


    # Create the model trainer
    dataloader_gen = DataLoaderGenerator(data, x_columns, y_columns, params.batch_size, params.data_loader_params)
    trainer = ModelTrainer(model, dataloader_gen, params.validation_method, params.criterion)
    if params.optimizer == "SGD":
        from torch.optim import SGD
        trainer.optimizer = SGD(model.parameters(), lr=params.lr, momentum=params.momentum, weight_decay=params.weight_decay)
    elif params.optimizer == "Adam":
        from torch.optim import Adam
        trainer.optimizer = Adam(model.parameters(), lr=params.lr)
    else:
        raise ValueError(f"Optimizer {params.optimizer} not recognized")
    
    # Train the model
    trainer.run(params.num_epochs)

    # Plot the losses
    if len(params.criterion.train_losses) == 1:
        fig_loss = plt.figure()
        params.criterion.loss_tracker.plot_losses()
    else:
        fig_loss, axes = plt.subplots(1, len(params.criterion.train_losses), squeeze=True, figsize=(17, 4), sharey=True)
        axes = axes.flatten()
        for i, loss_set in enumerate(zip(params.criterion.train_losses, params.criterion.val_losses)):
            plt.sca(axes[i])
            for loss in loss_set:
                plt.plot(params.criterion.loss_tracker.epoch_losses[loss], label=loss)
            plt.legend()
            plt.yscale('log')
        plt.show()


    # Change Batch Size for resolution in histograms
    trainer.set_batch_size(4096)

    # Get predictions
    train_error, train_pred = generate_model_error_and_prediction(trainer.model, trainer.train_loader, y_columns, y_scaler)
    val_error, val_pred = generate_model_error_and_prediction(trainer.model, trainer.validation_loader, y_columns, y_scaler)

    train_data = y_scaler.inverse_transform(trainer.train_loader.dataset[:][1].cpu())
    val_data = y_scaler.inverse_transform(trainer.validation_loader.dataset[:][1].cpu())
    
    fig_dist, axes = plt.subplots(3, len(y_columns), squeeze=True, figsize=(17, 8), sharey=True)
    if len(axes.shape) == 1:    # If only one column
        axes = axes.reshape(3, 1)

    for i in range(len(y_columns)): # Can change range and bins by changing the range param in hist instead of axis
        xlim = [min(train_data[:, i].min(), val_data[:, i].min()), max(train_data[:, i].max(), val_data[:, i].max())]
        # Plot Errors
        ax = axes[0, i]
        plt.sca(ax)
        column_name = train_error.columns[i]
        weights = np.ones(max(len(train_error[column_name]), len(val_error[column_name]))) / (len(train_error[column_name]) + len(val_error[column_name]))
        plt.hist(train_error[column_name], bins=params.bin_count, color='blue', alpha=0.5, label='Train', weights=weights[:len(train_error[column_name])])
        plt.hist(val_error[column_name], bins=params.bin_count, color='orange', alpha=0.5, label='Validation', weights=weights[:len(val_error[column_name])])
        axes[0, i].set_xlim([0, max(train_error[column_name].max(), val_error[column_name].max())])
        
        # Plot Predictions
        ax = axes[1, i]
        plt.sca(ax)
        column_name = train_pred.columns[i]
        weights = np.ones(max(len(train_pred[column_name]), len(val_pred[column_name]))) / (len(train_pred[column_name]) + len(val_pred[column_name]))
        plt.hist(train_pred[column_name], bins=params.bin_count, color='blue', alpha=0.5, label='Train', weights=weights[:len(train_pred[column_name])])
        plt.hist(val_pred[column_name], bins=params.bin_count, color='orange', alpha=0.5, label='Validation', weights=weights[:len(val_pred[column_name])])
        axes[1, i].set_xlim(xlim)
        
        # Plot Ground Truth
        ax = axes[2, i]
        plt.sca(ax)
        weights = np.ones(max(len(train_data[:, i]), len(val_data))) / (len(train_data[:, i]) + len(val_data[:, i]))
        plt.hist(train_data[:, i], bins=params.bin_count, color='blue', alpha=0.5, label='Train', weights=weights[:len(train_data[:, i])])
        plt.hist(val_data[:, i], bins=params.bin_count, color='orange', alpha=0.5, label='Validation', weights=weights[:len(val_data[:, i])])
        axes[2, i].set_xlim(xlim)

        # X Label for the bottommost row
        plt.xlabel(y_columns[i])

    # Y Labels
    axes_labels = ['MAE Error', 'Prediction', 'Ground Truth']
    for i in range(axes.shape[0]):
        axes[i, 0].set_ylabel(axes_labels[i])

    # Add labels to top-left subplot
    axes[0, 0].legend()
    plt.tight_layout()

    # Create loss string
    losses_str = ""
    for i, loss in enumerate(trainer.loss_func.loss_tracker.epoch_losses):
        if i % 2 == 0: # if even
            train_loss = trainer.loss_func.loss_tracker.epoch_losses[loss][-1]
        else:
            val_loss = trainer.loss_func.loss_tracker.epoch_losses[loss][-1]
            if len(loss.split("_")) > 2:
                losses_str += loss.split("_")[0]+f" | Train Loss: {train_loss}, Validation Loss: {val_loss}\n"
            else:
                losses_str += f"Train Loss: {train_loss}, Validation Loss: {val_loss}\n"
        

    # Save the report
    report = MarkdownReport(Path(f'results/{params.report_name.split("_")[-2]}'), params.report_name, params.report_title)
    report.add_code_report("Parameters", str(params))
    report.add_text_report("Objective", "Training Models using the ModelTrainer Class")
    report.add_code_report("Model Used", str(torchinfo.summary(trainer.model,verbose=0)))
    report.add_code_report("Model Properties", str(trainer))
    report.add_text_report("Loss Values", losses_str)
    report.add_image_report("Loss Curves", fig_loss)
    report.add_image_report("Error Distributions", fig_dist)
    report.save_report()


def check_args(args):
    params = Parameters()

    # Access the values of the arguments
    if hasattr(args, 'output_labels') and args.output_labels is not None:
        params.output_labels = args.output_labels
    if hasattr(args, 'subset_type') and args.subset_type is not None:
        params.subset_type = args.subset_type
    if hasattr(args, 'apply_log') and args.apply_log is not None:
        params.apply_log = args.apply_log
    if hasattr(args, 'random_seed') and args.random_seed is not None:
        params.random_seed = args.random_seed
    if hasattr(args, 'sample_size') and args.sample_size is not None:
        params.sample_size = args.sample_size
    if hasattr(args, 'test_size') and args.test_size is not None:
        params.test_size = args.test_size
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        params.batch_size = args.batch_size
    if hasattr(args, 'num_epochs') and args.num_epochs is not None:
        params.num_epochs = args.num_epochs
    if hasattr(args, 'model_type') and args.model_type is not None:
        params.set_model(args.model_type)
    if hasattr(args, 'dropout_type') and args.dropout_type is not None:
        params.set_dropout(args.dropout_type)
    if hasattr(args, 'validation_type') and args.validation_type is not None:
        params.set_validation(args.validation_type)
    if hasattr(args, 'criterion') and args.criterion is not None:
        params.set_criteria(args.criterion)
    if hasattr(args, 'optimizer') and args.optimizer is not None:
        params.optimizer = args.optimizer
    if hasattr(args, 'lr') and args.lr is not None:
        params.lr = args.lr
    if hasattr(args, 'momentum') and args.momentum is not None:
        params.momentum = args.momentum
    if hasattr(args, 'report_name') and args.report_name is not None:
        params.report_name = args.report_name
    if hasattr(args, 'report_title') and args.report_title is not None:
        params.report_title = args.report_title
    # if hasattr(args, 'hue') and args.hue is not None:
    #     params.set_hue(args.hue)

    return params


def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Description of your script')

    # Add arguments
    parser.add_argument('-ol', '--output_labels', type=str, help='Output labels')
    parser.add_argument('-st', '--subset_type', type=str, help='Subset type')
    parser.add_argument('-al', '--apply_log', type=bool, help='Apply log')
    parser.add_argument('-rs', '--random_seed', type=int, help='Random seed')
    parser.add_argument('-ss', '--sample_size', type=float, help='Sample size')
    parser.add_argument('-ts', '--test_size', type=float, help='Test size')
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size')
    parser.add_argument('-ne', '--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('-mt', '--model_type', type=int, help='Model type')
    parser.add_argument('-do', '--dropout_type', type=int, help='Dropout type')
    parser.add_argument('-vt', '--validation_type', type=str, help='Validation type')
    parser.add_argument('-c', '--criterion', type=str, help='Criterion')
    parser.add_argument('-o', '--optimizer', type=str, help='Optimizer')
    parser.add_argument('-l', '--lr', type=float, help='Learning rate')
    parser.add_argument('-m', '--momentum', type=float, help='Momentum')
    parser.add_argument('-rn', '--report_name', type=str, help='Report name')
    parser.add_argument('-rt', '--report_title', type=str, help='Report title')
    # parser.add_argument('-hue', '--hue', type=int, help='Hue')

    # Parse the arguments
    args, _ = parser.parse_known_args()
    
    # Check the arguments
    params = check_args(args)
    print(params)
    run(params)

if __name__ == '__main__':
    main()