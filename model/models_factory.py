class ModelsFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(model_name, opt):

        if model_name == 'ResNet34':
            from network.network import ResModel
            model = ResModel(opt.model_name, opt.num_classes)
        elif model_name == 'OAENet':
            from network.oaenet import OAENetModel
            model = OAENetModel(opt.model_name, opt.num_classes)
        else:
            raise ValueError("Model %s not recognized." % model_name)

        print("Model %s was created" % model.name)
        return model
