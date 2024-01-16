import timm




class CreateModel():
    def __init__(self, modelname, num_cls, pretrained=True):

        assert modelname in timm.list_models(pretrained=pretrained), 'Select a correct model name ==> see timm.list_models()'
        assert num_cls > 0, 'Number of Classes should be bigger than 0'
        
        self.num_cls = num_cls
        self.modelname = modelname
        self.pretrained= pretrained

    def load_model(self):
      self.model = timm.create_model(model_name=self.modelname,
                                num_classes=self.num_cls,
                                  pretrained=self.pretrained,
                                    in_chans=3)
      return self.model



