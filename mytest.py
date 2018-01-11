import glob
import moviepy.editor as mpy


path = 'dataset/Result/VOT/basketball/imgs'

gif_name = 'outputName'
fps = 12
file_list = glob.glob('dataset/Result/VOT/basketball/imgs/*.jpg') # Get all the pngs in the current directory
list.sort(file_list, key=lambda x: int(x.split('_')[0])) # Sort the images by #, this may need to be tweaked for your use case
clip = mpy.ImageSequenceClip(file_list, fps=fps)
clip.write_gif('{}.gif'.format(gif_name), fps=fps)




















# from PIL import Image
# import os
# from images2gif import writeGif
# filelist = []
# path = 'dataset/Result/VOT/basketball/imgs'
# files = os.listdir(path)
#
# for f in files:
#     if(os.path.isfile(path + '/' + f)):
#         if (os.path.splitext(f)[1] == ".jpg"):
#             filelist.append(f)
#
# iml = []
# for i in range(724):
#
#     name = '{}_predict.jpg'.format(i+1)
#     name = os.path.join(path, name)
#     img = Image.open(name)
#     iml.append(img)
#
# size = (600,350)
#
# for im in iml:
#     im.thumbnail(size, Image.ANTIALIAS)
#
#
# writeGif("fff.gif", iml, duration=0.5)
#
# # # for infile in filelist:
# # #   outfile = os.path.splitext(infile)[0] + ".gif"
# # #   if infile != outfile:
# # #     try:
# # #       Image.open(infile).save(outfile)
# # #       print "Covert to GIF successfully!"
# # #     except IOError:
# # #       print "This format can not support!", infile
#
#
# #
# # from images2gif import writeGif
# # import matplotlib.pyplot as plt
# # import numpy
# #
# # figure = plt.figure()
# # plot = figure.add_subplot(111)
# #
# # plot.hold(False)
# # # draw a cardinal sine plot
# # images = []
# # y = numpy.random.randn(100, 5)
# # for i in range(y.shape[1]):
# #     plot.plot(numpy.sin(y[:, i]))
# #     plot.set_ylim(-3.0, 3)
# #     plot.text(90, -2.5, str(i))
# #     im = Figtodat.fig2img(figure)
# #     images.append(im)
# #
# # writeGif("images.gif", images, duration=0.3, dither=0)