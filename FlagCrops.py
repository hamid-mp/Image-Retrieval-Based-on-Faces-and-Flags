from pathlib import Path
import cv2



class CropFlags():
    def __init__(self, root, output):
        self.paths = list(Path(root).glob('**/*.jpg'))
        self.output = Path(output)
        self.labels = list(Path(root).glob('**/*.txt'))
        assert len(self.paths) == len(self.labels)
    @staticmethod
    def label_analysis(txt_file, imW, imH):
        all_objects = []
        f =  open(txt_file, 'r')
        objects = f.readlines()

        for obj in objects:
            cls, x, y, w, h = obj.split(' ')
            x, y, w, h = float(x), float(y), float(w), float(h)
            x, y, w, h = int(x* imH), int(y* imW), int(w* imW), int(h* imH)
            x, y = x - w//2, y-h//2
            all_objects.append((cls, x, y, w, h))        
        return all_objects
    
    @staticmethod
    def read_image(path):
        image = cv2.imread(str(path))
        h, w, _ = image.shape
        return image, (h, w)
    
    def create_and_write(self, cls, image):
        cls = int(cls)
        dirname = f'{cls:03}'
        (self.output / dirname).mkdir(parents=True, exist_ok=True)

        count = sum(1 for x in (self.output / dirname).glob('*') if x.is_file())

        filename = f'{self.output / dirname / dirname}_{count:05}.jpg'
        cv2.imwrite(filename, image)


    def crop(self):

        for image, label in zip(self.paths, self.labels):
            try:
                image, size = self.read_image(image)
                
                objects = self.label_analysis(label, imW=size[1], imH=size[0])

                for flag in objects:
                    cls = flag[0]
                    x, y, w, h= flag[1:]
                    x = x if x >= 0  else 0
                    y = y if y >= 0  else 0
                    flag = image[y:y+h, x:x+w ]
                    
                    self.create_and_write(cls=cls, image = flag)
            except:
                continue
            
                
   

if __name__ == "__main__":

    images = CropFlags(r'Path To Dataset Images', output='Output Directory')
                       
    images.crop()
