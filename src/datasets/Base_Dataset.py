
class Base_Dataset_Class():
    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        """ Iterator for the dataset class, which will return the sample along with n randomly selected n-shot samples"""
        self.i += 1
        if self.i >= len(self.dataset.index):
            raise StopIteration

        if self.n_shot > 0:
            few_shot_samples = self.dataset.drop(self.dataset.index[self.i - 1]).sample(n=self.n_shot)
        else:
            few_shot_samples = None

        return self.dataset.index[self.i - 1], self.dataset.iloc[self.i - 1], few_shot_samples



    def set_n_shot(self, n_shot):
        self.n_shot = n_shot




