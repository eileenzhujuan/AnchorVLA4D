from abc import ABC, abstractmethod


class Converter(ABC):
    subclasses = []

    def __init_subclass__(cls, **kwargs):
        """All subclasses of Converter will be stored in the class attribute 'subclalsses'"""
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)

    @staticmethod
    @abstractmethod
    def hf_to_mm(cfg):
        pass

    @staticmethod
    @abstractmethod
    def mm_to_hf(cfg):
        pass

    @staticmethod
    @abstractmethod
    def resplit(cfg):
        pass
