
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']      # one input: L

    if model == 'plain':
        from models.model_plain import ModelPlain as M

    elif model == 'plain2':  # two inputs: L, C
        from models.model_plain2 import ModelPlain2 as M

    elif model == 'plain4':  # four inputs: L, k, sf, sigma
        from models.model_plain4 import ModelPlain4 as M

    elif model == 'gan':     # one input: L
        from models.model_gan import ModelGAN as M

    elif model == 'vrt':     # one video input L, for VRT
        from models.model_vrt import ModelVRT as M
    
    elif model == 'resnet101_sa':  # 假设这是你的模型标识符
        from models.model_resnet101_sa import ModelResNet101_SA as M

    elif model == 'srresnet':  
        from models.model_srresnet import ModelSRResNet as M
    
    elif model == 'srresnet2':  
        from models.model_srresnet2 import ModelSRResNet2 as M
    
    elif model == 'mem':  
        from models.model_mem import ModelMem as M
        
    elif model == 'unet':  
        from models.model_unet import ModelUNet as M
    
    elif model == 'catt':  
        from models.model_catt import ModelCAtt as M
    

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
