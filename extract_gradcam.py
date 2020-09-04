from scipy import ndimage
from skimage.measure import label, regionprops
from gradcam import GradCAM, GradCAMpp
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_bbox(cam, threshold):
    labeled, nr_objects = ndimage.label(cam > threshold)
    props = regionprops(labeled)
    return props

'''
def gradcam_area(target_model, img, model_type='vgg', layer_name='feature_29', threshold=0.8):
    img.unsqueeze_(0)
    configs = [
        dict(model_type=model_type, arch=target_model, layer_name=layer_name)
    ]

    for config in configs:
        config['arch'].to(device).eval()

    cams = [
        [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
        for config in configs
    ]
    
    for gradcam, gradcam_pp in cams:
        mask, _ = gradcam(img.cuda())
        
    cam = mask.cpu().numpy()
    props = generate_bbox(cam[0][0], threshold)
    
    return props
'''

def gradcam_area(target_model, target_layer, img, threshold=0.8):
    img.unsqueeze_(0)
    gradcam_model = GradCAM(target_model, target_layer)
    mask, _ = gradcam_model(img.cuda())
    
    cam = mask.cpu().numpy()
    props = generate_bbox(cam[0][0], threshold)
    
    return props