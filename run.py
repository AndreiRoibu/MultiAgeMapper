import os
import shutil
import argparse
import logging

import torch
import torch.utils.data as data
import numpy as np

from solver import Solver
from utils.data_utils import get_datasets_dynamically, get_test_datasets_dynamically
from utils.settings import Settings
import utils.data_evaluation as evaluations


# Set the default floating point tensor type to FloatTensor
torch.set_default_tensor_type(torch.FloatTensor)


def load_data_dynamically(data_parameters, mapping_evaluation_parameters=None, flag='train'):
    
    if flag=='train':
        print("Data is loading...")
        train_data, validation_data = get_datasets_dynamically(data_parameters)
        print("Data has loaded!")
        print("Training dataset size is {}".format(len(train_data)))
        print("Validation dataset size is {}".format(len(validation_data)))
        return train_data, validation_data
    elif flag=='test':
        print("Data is loading...")
        test_data, volumes_to_be_used, prediction_output_statistics_name = get_test_datasets_dynamically(data_parameters, mapping_evaluation_parameters)
        print("Data has loaded!")
        len_test_data = len(test_data)
        print("Testing dataset size is {}".format(len_test_data))
        return test_data, volumes_to_be_used, prediction_output_statistics_name, len_test_data
    else:
        print('ERROR: Invalid Flag')
        return None


def _load_pretrained_weights(model, number_of_modalities, previous_experiment_names, pretrained_model_directory,
                            save_model_directory,
                            freeze_pretrained_weights_flag, network_number,
                            ):
    
    previous_models = []
 
    assert len(previous_experiment_names) == number_of_modalities, "The number of modalities should be equal to the number of previous experiments!"

    for previous_experiment_name in previous_experiment_names:
        previous_models.append(torch.load(os.path.join('../' + pretrained_model_directory, save_model_directory, previous_experiment_name + '.pth.tar'),
                                    map_location='cpu')
                            )

    for idx, previous_model in enumerate(previous_models):
        if hasattr(previous_model, 'state_dict'):
            previous_models[idx] = previous_model.state_dict()

    model_state_dict = model.state_dict()


    if network_number == 3:
        specific_path = 'Convolution_Paths'
    elif network_number == 5:
        specific_path = 'Modality_Paths'
        # fully_connected_name = 'FullyConnectedModality'
    else:
        specific_path = 'Modality_Paths'
        # fully_connected_name = 'FullyConnectedModality'
        print('NETWORK NUMBER MUST BE 3 or 5! Other versions not yet coded / tried! ERRORS LIKELY!')

    fully_connected_name = 'FullyConnectedModality'
    # print(fully_connected_name)

    for idx in range(number_of_modalities):

        path_name = 'Path_' + str(idx) + '_'
        merging_paths_name = specific_path + '.' + str(idx) + '.'

        for key in model_state_dict.keys():
            if key.startswith(merging_paths_name):
                key_stripped = key.replace(merging_paths_name, '')
                key_stripped = key_stripped.replace(path_name, '')
                if key_stripped.startswith(fully_connected_name):
                    key_stripped = key_stripped.replace(fully_connected_name, 'FullyConnected')
                model_state_dict.update({key : previous_models[idx][key_stripped]})

    model.load_state_dict(model_state_dict)
    print("Pretrained models loaded successfully!")

    if freeze_pretrained_weights_flag == True:
        
        pre_trained_parameters_counter = 0
        for i in model.named_parameters():
            if i[0].startswith(specific_path):
                pre_trained_parameters_counter += 1

        param_counter = 0
        for param in model.parameters():
            if param_counter < pre_trained_parameters_counter:
                param.requires_grad = False
            else:
                break
            param_counter += 1

    print('\n') 
    print('Total number of model parameters')
    print(sum([p.numel() for p in model.parameters()]))
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('Total number of trainable parameters')
    params = sum([p.numel() for p in model_parameters])
    print(params)
    print('\n')

    return model


def train(data_parameters, training_parameters, network_parameters, misc_parameters):

    if training_parameters['optimiser'] == 'adamW':
        optimizer = torch.optim.AdamW
    elif training_parameters['optimiser'] == 'adam':
        optimizer = torch.optim.Adam
    else:
        optimizer = torch.optim.Adam # Default option

    optimizer_arguments={'lr': training_parameters['learning_rate'],
                        'betas': training_parameters['optimizer_beta'],
                        'eps': training_parameters['optimizer_epsilon'],
                        'weight_decay': training_parameters['optimizer_weigth_decay']
                        }

    if training_parameters['loss_function'] == 'mse':
        loss_function = torch.nn.MSELoss()
    elif training_parameters['loss_function'] == 'mae':
        loss_function = torch.nn.L1Loss()
    else:
        print("Loss function not valid. Defaulting to MSE!")
        loss_function = torch.nn.MSELoss(reduction='batchmean')

    train_data, validation_data = load_data_dynamically(data_parameters=data_parameters, flag='train')
    train_loader = data.DataLoader(
        dataset=train_data,
        batch_size=training_parameters['training_batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=data_parameters['num_workers']
    )
    validation_loader = data.DataLoader(
        dataset=validation_data,
        batch_size=training_parameters['validation_batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=data_parameters['num_workers']
    )

    number_of_modalities = int(len(data_parameters['modality_flag']))

    if data_parameters['t1t2ratio_flag'] == 2:
        number_of_modalities += 1

    if network_parameters['network_number'] == 1 or network_parameters['network_number']==2:
        data_parameters['fused_data_flag'] = True
    else:
        data_parameters['fused_data_flag'] = False
    
    if data_parameters['fused_data_flag'] == True:
        original_input_channels = number_of_modalities
    else:
        original_input_channels = 1

    # if training_parameters['use_pre_trained']:
    #     pre_trained_path = "saved_models/" + training_parameters['pre_trained_experiment_name'] + ".pth.tar"
    #     AgeMapperModel = torch.load(pre_trained_path, map_location=torch.device('cpu'))
    #     print('--> Using PRE-PRAINED NETWORK: ', pre_trained_path)
    # else:
    if network_parameters['network_number'] == 1:
        from MultiAgeMapper import AgeMapper_input1
        AgeMapperModel = AgeMapper_input1(
                                    fused_data_flag=data_parameters['fused_data_flag'],
                                    original_input_channels = original_input_channels,
                                    network_parameters=network_parameters
                                    )
        print('--> Using NETWORK 1')
    elif network_parameters['network_number'] == 2:
        from MultiAgeMapper import AgeMapper_input2
        AgeMapperModel = AgeMapper_input2(
                                    fused_data_flag=data_parameters['fused_data_flag'],
                                    original_input_channels = original_input_channels,
                                    network_2_modality_filter_outputs = network_parameters['network_2_modality_filter_outputs']
                                    )
        print('--> Using NETWORK 2')
    elif network_parameters['network_number'] == 3:
        from MultiAgeMapper import AgeMapper_input3
        AgeMapperModel = AgeMapper_input3(
                                    fused_data_flag=data_parameters['fused_data_flag'],
                                    original_input_channels = number_of_modalities,
                                    network_parameters=network_parameters
                                    )
        print('--> Using NETWORK 3')
    elif network_parameters['network_number'] == 4:
        from MultiAgeMapper import AgeMapper_input4
        AgeMapperModel = AgeMapper_input4(
                                    fused_data_flag=data_parameters['fused_data_flag'],
                                    original_input_channels = number_of_modalities
                                    )
        print('--> Using NETWORK 4')
    elif network_parameters['network_number'] == 5:
        from MultiAgeMapper import AgeMapper_input5
        AgeMapperModel = AgeMapper_input5(
                                    fused_data_flag=data_parameters['fused_data_flag'],
                                    original_input_channels = number_of_modalities,
                                    network_parameters=network_parameters
                                    )
        print('--> Using NETWORK 5')
    elif network_parameters['network_number'] == 6:
        from MultiAgeMapper import AgeMapper_input6
        AgeMapperModel = AgeMapper_input6(
                                    fused_data_flag=data_parameters['fused_data_flag'],
                                    original_input_channels = number_of_modalities
                                    )
        print('--> Using NETWORK 6')
    elif network_parameters['network_number'] == 7:
        from MultiAgeMapper import AgeMapper_input7
        AgeMapperModel = AgeMapper_input7(
                                    fused_data_flag=data_parameters['fused_data_flag'],
                                    original_input_channels = number_of_modalities
                                    )
        print('--> Using NETWORK 7')
    elif network_parameters['network_number'] == 8:
        from MultiAgeMapper import AgeMapper_input8
        AgeMapperModel = AgeMapper_input8(
                                    fused_data_flag=data_parameters['fused_data_flag'],
                                    original_input_channels = number_of_modalities
                                    )
        print('--> Using NETWORK 8')
    elif network_parameters['network_number'] == 9:
        from MultiAgeMapper import AgeMapper_input9
        AgeMapperModel = AgeMapper_input9(
                                    fused_data_flag=data_parameters['fused_data_flag'],
                                    original_input_channels = number_of_modalities
                                    )
        print('--> Using NETWORK 9')

    if training_parameters['use_pre_trained']:
        pre_trained_path = "saved_models/" + training_parameters['pre_trained_experiment_name'] + ".pth.tar"
        AgeMapperModel_pretrained = torch.load(pre_trained_path, map_location=torch.device('cpu'))
        AgeMapperModel.load_state_dict(AgeMapperModel_pretrained)
        del AgeMapperModel_pretrained
        print('--> Using PRE-TRAINED NETWORK: ', pre_trained_path)
        print('\n') 
        print('Total number of model parameters')
        print(sum([p.numel() for p in AgeMapperModel.parameters()]))
        model_parameters = filter(lambda p: p.requires_grad, AgeMapperModel.parameters())
        print('Total number of trainable parameters')
        params = sum([p.numel() for p in model_parameters])
        print(params)
        print('\n')

    if network_parameters['use_transfer_learning'] == True:
        AgeMapperModel = _load_pretrained_weights(model=AgeMapperModel,
                                                 number_of_modalities = number_of_modalities, 
                                                previous_experiment_names = network_parameters['previous_experiment_names'], 
                                                pretrained_model_directory = network_parameters['pretrained_model_directory'],
                                                save_model_directory = misc_parameters['save_model_directory'],
                                                freeze_pretrained_weights_flag = network_parameters['freeze_pretrained_weights_flag'],
                                                network_number = network_parameters['network_number']
                                                )
        print('--> Using TRANSFER LEARNING NETWORKs: ', network_parameters['previous_experiment_names'])


    solver = Solver(model=AgeMapperModel,
                    number_of_classes=network_parameters['number_of_classes'],
                    experiment_name=training_parameters['experiment_name'],
                    optimizer=optimizer,
                    optimizer_arguments=optimizer_arguments,
                    loss_function=loss_function,
                    model_name=training_parameters['experiment_name'],
                    number_epochs=training_parameters['number_of_epochs'],
                    loss_log_period=training_parameters['loss_log_period'],
                    learning_rate_scheduler_step_size=training_parameters['learning_rate_scheduler_step_size'],
                    learning_rate_scheduler_gamma=training_parameters['learning_rate_scheduler_gamma'],
                    use_last_checkpoint=training_parameters['use_last_checkpoint'],
                    experiment_directory=misc_parameters['experiments_directory'],
                    logs_directory=misc_parameters['logs_directory'],
                    checkpoint_directory=misc_parameters['checkpoint_directory'],
                    best_checkpoint_directory=misc_parameters['best_checkpoint_directory'],
                    save_model_directory=misc_parameters['save_model_directory'],
                    learning_rate_validation_scheduler=training_parameters['learning_rate_validation_scheduler'],
                    learning_rate_cyclical = training_parameters['learning_rate_cyclical'],
                    learning_rate_scheduler_patience=training_parameters['learning_rate_scheduler_patience'],
                    learning_rate_scheduler_threshold=training_parameters['learning_rate_scheduler_threshold'],
                    learning_rate_scheduler_min_value=training_parameters['learning_rate_scheduler_min_value'],
                    learning_rate_scheduler_max_value=training_parameters['learning_rate_scheduler_max_value'],
                    learning_rate_scheduler_step_number=training_parameters['learning_rate_scheduler_step_number'],
                    early_stopping_patience=training_parameters['early_stopping_patience'],
                    early_stopping_min_patience=training_parameters['early_stopping_min_patience'],
                    early_stopping_min_delta=training_parameters['early_stopping_min_delta'],
                    fused_data_flag = data_parameters['fused_data_flag'],
                    use_transfer_learning = network_parameters['use_transfer_learning'],
                    use_pre_trained = training_parameters['use_pre_trained'],
                    )

    solver.train(train_loader, validation_loader)

    del train_data, validation_data, train_loader, validation_loader, AgeMapperModel, solver, optimizer
    torch.cuda.empty_cache()


def evaluate_data(mapping_evaluation_parameters, data_parameters, network_parameters):

    test_data, volumes_to_be_used, prediction_output_statistics_name, len_test_data = load_data_dynamically(
                                                                                                            data_parameters=data_parameters, 
                                                                                                            mapping_evaluation_parameters=mapping_evaluation_parameters, 
                                                                                                            flag='test'
                                                                                                            )

    test_loader = data.DataLoader(
        dataset = test_data,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=data_parameters['num_workers']
    )

    number_of_modalities = int(len(data_parameters['modality_flag']))

    if data_parameters['t1t2ratio_flag'] == 2:
        number_of_modalities += 1

    if network_parameters['network_number'] == 1 or network_parameters['network_number']==2:
        data_parameters['fused_data_flag'] = True
    else:
        data_parameters['fused_data_flag'] = False

    if data_parameters['fused_data_flag'] == True:
        original_input_channels = number_of_modalities
    else:
        original_input_channels = 1

    fused_data_flag=data_parameters['fused_data_flag']

    if network_parameters['network_number'] == 1:
        from MultiAgeMapper import AgeMapper_input1
        AgeMapperModel = AgeMapper_input1(
                                    fused_data_flag=fused_data_flag,
                                    original_input_channels = original_input_channels,
                                    network_parameters=network_parameters
                                    )
    elif network_parameters['network_number'] == 2:
        from MultiAgeMapper import AgeMapper_input2
        AgeMapperModel = AgeMapper_input2(
                                    fused_data_flag=fused_data_flag,
                                    original_input_channels = original_input_channels,
                                    network_2_modality_filter_outputs = network_parameters['network_2_modality_filter_outputs']
                                    )
    elif network_parameters['network_number'] == 3:
        from MultiAgeMapper import AgeMapper_input3
        AgeMapperModel = AgeMapper_input3(
                                    fused_data_flag=fused_data_flag,
                                    original_input_channels = number_of_modalities,
                                    network_parameters=network_parameters
                                    )
    elif network_parameters['network_number'] == 4:
        from MultiAgeMapper import AgeMapper_input4
        AgeMapperModel = AgeMapper_input4(
                                    fused_data_flag=data_parameters['fused_data_flag'],
                                    original_input_channels = number_of_modalities
                                    )
        print('--> Using NETWORK 4')
    elif network_parameters['network_number'] == 5:
        from MultiAgeMapper import AgeMapper_input5
        AgeMapperModel = AgeMapper_input5(
                                    fused_data_flag=data_parameters['fused_data_flag'],
                                    original_input_channels = number_of_modalities,
                                    network_parameters=network_parameters
                                    )
        print('--> Using NETWORK 5')
    elif network_parameters['network_number'] == 6:
        from MultiAgeMapper import AgeMapper_input6
        AgeMapperModel = AgeMapper_input6(
                                    fused_data_flag=data_parameters['fused_data_flag'],
                                    original_input_channels = number_of_modalities
                                    )
        print('--> Using NETWORK 6')
    elif network_parameters['network_number'] == 7:
        from MultiAgeMapper import AgeMapper_input7
        AgeMapperModel = AgeMapper_input7(
                                    fused_data_flag=data_parameters['fused_data_flag'],
                                    original_input_channels = number_of_modalities
                                    )
        print('--> Using NETWORK 7')
    elif network_parameters['network_number'] == 8:
        from MultiAgeMapper import AgeMapper_input8
        AgeMapperModel = AgeMapper_input8(
                                    fused_data_flag=data_parameters['fused_data_flag'],
                                    original_input_channels = number_of_modalities
                                    )
        print('--> Using NETWORK 8')
    elif network_parameters['network_number'] == 9:
        from MultiAgeMapper import AgeMapper_input9
        AgeMapperModel = AgeMapper_input9(
                                    fused_data_flag=data_parameters['fused_data_flag'],
                                    original_input_channels = number_of_modalities
                                    )
        print('--> Using NETWORK 9')

    device = mapping_evaluation_parameters['device']

    experiment_name = mapping_evaluation_parameters['experiment_name']
    trained_model_path = "saved_models/" + experiment_name + ".pth.tar"
    prediction_output_path = experiment_name + "_predictions"
    control = mapping_evaluation_parameters['control']
    dataset_sex = data_parameters['dataset_sex']
    
    evaluations.evaluate_data(
                        model = AgeMapperModel,
                        test_loader = test_loader,
                        volumes_to_be_used = volumes_to_be_used, 
                        prediction_output_statistics_name = prediction_output_statistics_name, 
                        trained_model_path = trained_model_path,
                        device = device,
                        prediction_output_path = prediction_output_path,
                        control = control,
                        fused_data_flag = fused_data_flag,
                        dataset_sex = dataset_sex,
                        len_test_data = len_test_data,
                    )


def delete_files(folder):
    for object_name in os.listdir(folder):
        file_path = os.path.join(folder, object_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as exception:
            print(exception)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', required=True,
                        help='run mode, valid values are train, evaluate-data, clear-checkpoints, clear-checkpoints-completely, clear-logs, clear-experiment, clear-experiment-completely, train-and-evaluate-mapping, lr-range-test, solver-logger-test')
    parser.add_argument('--model_name', '-n', required=True,
                        help='model name, required for identifying the settings file modelName.ini & modelName_eval.ini')
    parser.add_argument('--use_last_checkpoint', '-c', required=False,
                        help='flag indicating if the last checkpoint should be used if 1; useful when wanting to time-limit jobs.')
    parser.add_argument('--number_of_epochs', '-e', required=False,
                        help='flag indicating how many epochs the network will train for; should be limited to ~3 hours or 2/3 epochs')

    arguments = parser.parse_args()

    settings_file_name = arguments.model_name + '.ini'
    evaluation_settings_file_name = arguments.model_name + '_eval.ini'

    settings = Settings(settings_file_name)
    data_parameters = settings['DATA']
    training_parameters = settings['TRAINING']
    network_parameters = settings['NETWORK']
    misc_parameters = settings['MISC']

    if arguments.use_last_checkpoint == '1':
        training_parameters['use_last_checkpoint'] = True
    elif arguments.use_last_checkpoint == '0':
        training_parameters['use_last_checkpoint'] = False

    if arguments.number_of_epochs is not None:
        training_parameters['number_of_epochs'] = int(arguments.number_of_epochs)

    if arguments.mode == 'train':
        train(data_parameters, training_parameters, network_parameters, misc_parameters)

    elif arguments.mode == 'evaluate-data':
        logging.basicConfig(filename='evaluate-data-error.log')
        settings_evaluation = Settings(evaluation_settings_file_name)
        mapping_evaluation_parameters = settings_evaluation['MAPPING']
        evaluate_data(mapping_evaluation_parameters, data_parameters, network_parameters)

    elif arguments.mode == 'clear-checkpoints':

        warning_message = input("Warning! This command will delete all checkpoints. Continue [y]/n: ")
        if warning_message == 'y':
            if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory'])):
                shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory']))
                print('Cleared the current experiment checkpoints successfully!')
            else:
                print('ERROR: Could not find the experiment checkpoints.')
        else:
            print("Action Cancelled!")

    elif arguments.mode == 'clear-checkpoints-completely':
        warning_message = input("WARNING! This command will delete all checkpoints (INCL BEST). DANGER! Continue [y]/n: ")
        if warning_message == 'y':
            warning_message2 = input("ARE YOU SURE? [y]/n: ")
            if warning_message2 == 'y':
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory']))
                    print('Cleared the current experiment checkpoints successfully!')
                else:
                    print('ERROR: Could not find the experiment checkpoints.')
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['best_checkpoint_directory'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['best_checkpoint_directory']))
                    print('Cleared the current experiment best checkpoints successfully!')
                else:
                    print('ERROR: Could not find the experiment best checkpoints.')
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name']))
                    print('Cleared the current experiment folder successfully!')
                else:
                    print("ERROR: Could not find the experiment folder.")
            else:
                print("Action Cancelled!")
        else:
            print("Action Cancelled!")

    elif arguments.mode == 'clear-logs':

        warning_message = input("Warning! This command will delete all checkpoints and logs. Continue [y]/n: ")
        if warning_message == 'y':
            if os.path.exists(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name'])):
                shutil.rmtree(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name']))
                print('Cleared the current experiment logs directory successfully!')
            else:
                print("ERROR: Could not find the experiment logs directory!")
        else:
            print("Action Cancelled!")

    elif arguments.mode == 'clear-experiment':

        warning_message = input("Warning! This command will delete all checkpoints and logs. Continue [y]/n: ")
        if warning_message == 'y':
            if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory'])):
                shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory']))
                print('Cleared the current experiment checkpoints successfully!')
            else:
                print('ERROR: Could not find the experiment checkpoints.')
            if os.path.exists(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name'])):
                shutil.rmtree(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name']))
                print('Cleared the current experiment logs directory successfully!')
            else:
                print("ERROR: Could not find the experiment logs directory!")
        else:
            print("Action Cancelled!")

    elif arguments.mode == 'clear-experiment-completely':
        warning_message = input("WARNING! This command will delete all checkpoints (INCL BEST) and logs. DANGER! Continue [y]/n: ")
        if warning_message == 'y':
            warning_message2 = input("ARE YOU SURE? [y]/n: ")
            if warning_message2 == 'y':
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['checkpoint_directory']))
                    print('Cleared the current experiment checkpoints successfully!')
                else:
                    print('ERROR: Could not find the experiment checkpoints.')
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['best_checkpoint_directory'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'], misc_parameters['best_checkpoint_directory']))
                    print('Cleared the current experiment best checkpoints successfully!')
                else:
                    print('ERROR: Could not find the experiment best checkpoints.')
                if os.path.exists(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name'])):
                    shutil.rmtree(os.path.join(misc_parameters['experiments_directory'], training_parameters['experiment_name']))
                    print('Cleared the current experiment folder successfully!')
                else:
                    print("ERROR: Could not find the experiment folder.")
                if os.path.exists(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name'])):
                    shutil.rmtree(os.path.join(misc_parameters['logs_directory'], training_parameters['experiment_name']))
                    print('Cleared the current experiment logs directory successfully!')
                else:
                    print("ERROR: Could not find the experiment logs directory!")
            else:
                print("Action Cancelled!")
        else:
            print("Action Cancelled!")

    # elif arguments.mode == 'clear-everything':
    #     delete_files(misc_parameters['experiments_directory'])
    #     delete_files(misc_parameters['logs_directory'])
    #     print('Cleared the all the checkpoints and logs directory successfully!')

    elif arguments.mode == 'train-and-evaluate-data':
        settings_evaluation = Settings(evaluation_settings_file_name)
        mapping_evaluation_parameters = settings_evaluation['MAPPING']
        train(data_parameters, training_parameters,
              network_parameters, misc_parameters)
        logging.basicConfig(filename='evaluate-mapping-error.log')
        evaluate_data(mapping_evaluation_parameters)
      
    else:
        raise ValueError('Invalid mode value! Only supports: train, evaluate-data, evaluate-mapping, train-and-evaluate-mapping, clear-checkpoints, clear-logs,  clear-experiment and clear-everything (req uncomment for safety!)')
