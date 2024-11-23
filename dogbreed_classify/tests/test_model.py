import torch
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def test_model_forward(model):
    batch_size = 4
    channels = 3
    height, width = 224, 224
    
    x = torch.randn(batch_size, channels, height, width)
    output = model(x)
    
    assert output.shape == (batch_size, model.hparams.num_classes)

def test_model_training_step(model):
    batch_size = 4
    channels = 3
    height, width = 224, 224
    
    x = torch.randn(batch_size, channels, height, width)
    y = torch.randint(0, model.hparams.num_classes, (batch_size,))
    
    loss = model.training_step((x, y), 0)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()

def test_model_validation_step(model):
    batch_size = 4
    channels = 3
    height, width = 224, 224
    
    x = torch.randn(batch_size, channels, height, width)
    y = torch.randint(0, model.hparams.num_classes, (batch_size,))
    
    model.validation_step((x, y), 0)
    
    assert model.val_acc.compute() >= 0 and model.val_acc.compute() <= 1

def test_model_test_step(model):
    batch_size = 4
    channels = 3
    height, width = 224, 224
    
    x = torch.randn(batch_size, channels, height, width)
    y = torch.randint(0, model.hparams.num_classes, (batch_size,))
    
    model.test_step((x, y), 0)
    
    assert model.test_acc.compute() >= 0 and model.test_acc.compute() <= 1