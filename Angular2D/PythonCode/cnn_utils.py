"""Handles functions such as loading from checkpoints"""
import os
import math

import torch
import torch.nn as nn

def check_cuda(config):
    """Checks cuda settings from config - Returns true if cuda available"""
    cuda = config['NETWORK']['cuda'] == 'True'
    cuda_device = config['NETWORK']['gpu_id']
    if cuda:
        print("=> using gpu id: '{}'".format(cuda_device))
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id")
        print("=> cudnn version is", torch.backends.cudnn.version())
    return cuda

def save_checkpoint(model, epoch, optimizer, best_loss, save_dir, name):
    """Saves model params and epoch number at save_dir/name"""
    if (model is None) or (epoch is None):
        print("No model or epoch given for saving")
        return -1
        
    model_out_path = os.path.join(save_dir, name)
    state = {"epoch": epoch, 
             "model": model,
             "state_dict": model.state_dict(),
             "best_loss": best_loss,
             "optimizer": optimizer.state_dict()}

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

def load_from_checkpoint(model, optimizer, args, config):
    resume_location = os.path.join(
        config['PATH']['checkpoint_dir'],
        args.checkpoint)
    best_loss = math.inf
    if os.path.isfile(resume_location):
        print("=> loading checkpoint '{}'".format(resume_location))
        checkpoint = torch.load(resume_location)
        args.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"].state_dict(), False, args.first)
        best_loss = checkpoint["best_loss"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("=> loaded checkpoint '{}' (epoch {})".format(
                resume_location, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_location))
    return best_loss

def load_weights(model, args, config, frozen=False):
    """Load the state dict into the passed in model"""

    weights_location = os.path.join(
        config['PATH']['model_dir'],
        args.pretrained)
    if os.path.isfile(weights_location):
        print("=> loading model '{}'".format(weights_location))
        weights = torch.load(weights_location)
        if 'model' in weights:
            model.load_state_dict(weights['model'].state_dict(), frozen, args.first)
        else:
            model.load_state_dict(weights, frozen, args.first)
    else:
        print("=> no model found at '{}'".format(weights_location))

def load_model_and_weights(args, config):
    """Load the model defined in config and args and the weights"""
    # Load the architecture
    model = torch.load(os.path.join(
                config['PATH']['model_dir'], args.pretrained))['model']
    # Load the weights
    load_weights(model, args, config)
    return model

def print_mem_usage():
    """Prints torch gpu memeory usage"""
    print("Using {:5} / {:5} tensor memory {:.1f}% of max".format(
        torch.cuda.memory_allocated(),
        torch.cuda.max_memory_allocated(),
        (torch.cuda.memory_allocated() / 
         torch.cuda.max_memory_allocated()) * 100
    ))
    print("Using {} / {} cached memory {:.1f}% of max".format(
        torch.cuda.memory_cached(),
        torch.cuda.max_memory_cached(),
        (torch.cuda.memory_cached() / 
         torch.cuda.max_memory_cached()) * 100
    ))

def log_all_layer_weights(model, writer, epoch):
    log_layer_weights(
        model, writer, epoch,
        'body.0', 'First layer weight')
    log_layer_weights(
        model, writer, epoch,
        'body.1', 'Second layer weight')
    log_layer_weights(
        model, writer, epoch,
        'body.2', 'Third layer weight')
    log_layer_weights(
        model, writer, epoch,
        'body.3', 'Last layer weight')

def log_layer_weights(model, writer, epoch, name, out_string):
    name = name + '.body.0.weight'
    layer_weights = model.state_dict()[name]
    writer.add_histogram(
        out_string, layer_weights, epoch, 'auto')

def transform_lf_to_torch(lf):
    """
    Torch expects a grid of images to be in the format:
    (B x C x H x W) but our light field grids are in the format:
    (B x W x H x C) so need to transpose them
    """
    return lf.transpose(1, 3).transpose(2, 3)

def check_gradients(model):
    """Returns false if the model has no gradients at the first level"""
    gradient_params = list(
        filter(lambda p: p.requires_grad, model.parameters()))
    if gradient_params[0].grad is None:
        print("There is some bad error, gradients are None!")
        return False
    return True

def log_children(model):
    child_counter = 0
    for child in model.children():
        print(" child", child_counter, "is -")
        print(child)
        child_counter += 1

def log_child_gradients(model):
    print("\nPrinting model information")
    for child_name, child in model.named_children():
        print(" child", child_name, "is -")
        print(child)
        for name, param in child.named_parameters():
            print(
                "Parameter {} requires grad?".format(name), 
                param.requires_grad)