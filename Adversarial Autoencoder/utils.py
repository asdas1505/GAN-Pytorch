import os 
from PIL import Image 
import matplotlib.pyplot as plt
from torchvision.utils import save_image


def make_gif(images_path, save_path):
    images_dir = images_path
    image_list = []
    for i in range(len(os.listdir(images_dir))):
      temp_img = Image.open(images_dir + '/sample_' + str(i) + '.png' )
      image_list.append(temp_img)
    image_list[0].save(save_path + 'out.gif', save_all=True, append_images=image_list[1:])


def latent_space_visualization(data_iter, save=False):

	images, labels = next(iter(data_iter))
	z_plot = encoder(images)
	plt.figure(figsize=(12,12))
	plt.scatter(z_plot.detach().numpy()[:,0], z_plot.detach().numpy()[:,1],c = labels.detach().numpy(), cmap = 'Set1')
	if save:
		plt.savefig('MNIST_latent_space.png')
	plt.show()


def reconstruct_images(images, epoch):
    
    encoder.eval()
    decoder.eval()
    
    z_save = encoder(images) 
    sample = decoder(z_save)
    
    save_image(sample.view(images.shape[0], 1, 28, 28), 'samples/sample_' + str(epoch) + '.png')