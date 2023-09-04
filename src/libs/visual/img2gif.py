import matplotlib.pyplot as plt
import imageio
from IPython.display import Image
from google.colab import output

def slice_vol(volumes, channel=1, full= False):
  """
  Input: 
    - volumes: pytorch tensor 4d channel
    - channel: channel you want to show --> (int)
    - full: if you want to show all channel --> (boolen)
  Output:
    - plot all sclice of your volumes
  """
  col = 10
  row = int(volumes.shape[1]/col) +2
  # print(col, row)
  plt.figure(figsize=(20,30))
  if full:
    volumes = volumes.permute(0,3,1,2)
    for idx, img in enumerate(volumes):
      # print(img.shape)
      plt.subplot(row,col,idx+1)
      plt.imshow((0.1*img).astype("uint8"))
  else:
    volumes = volumes.permute(0,3,1,2)
    for idx, img in enumerate(volumes[channel]):
      plt.subplot(col,row,idx+1)
      plt.imshow(img)


def save_gif(vol, pathname, full= False):
  """
  Input: 
    - vol: pytorch tensor 4d channel with size (c x d x h x w)
    - pathname: path of your save file --> (string)
    - full: if you want to show all channel --> (boolen)
  Output:
    - save your volumes in gif
  """
  if full:
    vol = vol.permute(3,1,2,0)
    imageio.mimsave(f'{pathname}.gif', vol, format = 'GIF-PIL', fps = 20)
    output.clear()
    print(f"{pathname}.gif ===> Done!")
  else:
    vol = vol.permute(0,3,1,2)
    for i in range(4):
      imageio.mimsave(f'{pathname}_{i}.gif', vol[i], format = 'GIF-PIL', fps = 20)
      output.clear()
      print(f'{pathname}_{i}.gif ===> Done!')
      
def show_gif(pathname):
  """
  Input: 
    - pathname: path of file you want to show --> (string)
  Output:
    - save your volumes in gif
  """
  return Image(open(pathname, 'rb').read())