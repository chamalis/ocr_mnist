import cPickle, sys, os, argparse, copy
import numpy as np
from PIL import Image

SIZE = 14  #2-D size, the classifier was trained
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify a single number')    
    parser.add_argument('-n', '--ann', help='give filename where the trained ANN has been stored. Required', required=True)
    parser.add_argument('-p', '--picture',  help='give file path containing picture. Required', required=True)
    parser.add_argument('-fc', '--foreground',  help='give foreground color (B/W), default:B', default='B')
    args = parser.parse_args()
    
    im = Image.open(args.picture).convert('L')
    if (np.shape(im)[0]) != SIZE or (np.shape(im)[1]) != SIZE :
        im = im.resize( (SIZE*2, SIZE*2) )
        im = im.resize( (SIZE,SIZE), Image.ANTIALIAS )
    #im.show()
    #print np.shape(im)
    picture = np.asarray(im)
    #print picture
    if args.foreground == 'B':  #classifier was trained with white foreground so, reverse the image intensities if black
        picture = picture.max() - picture
    
    print picture 
    img = Image.fromarray(picture)
    img.show()
    
    picture = picture.reshape(SIZE**2,)  #reshape from 14x14 to 196x1
    picture = (picture - picture.mean() ) / picture.max()   #normalize intensities to [-1 , 1]
    #print picture.max(), picture.min()
    #print np.shape(picture)   
    
    #load the trained classifier
    f = file(args.ann, 'rb')
    net = cPickle.load(f)
    f.close() 
   
    oneNenc = net.produce_output(picture)  #run the classifier
    print oneNenc
    
    print '\n *** The number was classified as >>> ' + str(oneNenc.argmax()) + ' <<< ***'
