import os
import numpy as np
from gin_train.utils import write_json, load_pickle
from tqdm import tqdm
from kipoi.data_utils import numpy_collate_concat
from kipoi.external.flatten_json import flatten
import gin
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@gin.configurable
class KerasTrainer:
    """Simple Keras model trainer
    """

    def __init__(self, model, train_dataset, valid_dataset, output_dir, cometml_experiment=None):
        """
        Args:
          model: compiled keras.Model
          train: training Dataset (object inheriting from kipoi.data.Dataset)
          valid: validation Dataset (object inheriting from kipoi.data.Dataset)
          output_dir: output directory where to log the training
          cometml_experiment: if not None, append logs to commetml
        """
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.cometml_experiment = cometml_experiment

        # setup the output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ckp_file = f"{self.output_dir}/model.h5"
        if os.path.exists(self.ckp_file):
            raise ValueError(f"model.h5 already exists in {self.output_dir}")
        self.history_path = f"{self.output_dir}/history.csv"
        self.evaluation_path = f"{self.output_dir}/evaluation.valid.json"

    def train(self,
              batch_size=256,
              epochs=100,
              early_stop_patience=4,
              num_workers=8,
              train_epoch_frac=1.0,
              valid_epoch_frac=1.0,
              train_batch_sampler=None,
              tensorboard=True):
        """Train the model
        Args:
          batch_size:
          epochs:
          patience: early stopping patience
          num_workers: how many workers to use in parallel
          train_epoch_frac: if smaller than 1, then make the epoch shorter
          valid_epoch_frac: same as train_epoch_frac for the validation dataset
          train_batch_sampler: batch Sampler for training. Useful for say Stratified sampling
          tensorboard: if True, tensorboard output will be added
        """
        from keras.callbacks import EarlyStopping, History, CSVLogger, ModelCheckpoint, TensorBoard
        from keras.models import load_model

        if train_batch_sampler is not None:
            train_it = self.train_dataset.batch_train_iter(shuffle=False,
                                                           batch_size=1,
                                                           drop_last=None,
                                                           batch_sampler=train_batch_sampler,
                                                           num_workers=num_workers)
        else:
            train_it = self.train_dataset.batch_train_iter(batch_size=batch_size,
                                                           shuffle=True,
                                                           num_workers=num_workers)
        print("Got training iterator. Batch size:", batch_size, "num_workers:", num_workers, 
              "Train batch sampler:", train_batch_sampler)
        next(train_it)
        logger.info("Got training set")
        valid_it = self.valid_dataset.batch_train_iter(batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=num_workers)
        logger.info("Got validation iterator")
        next(valid_it)
        logger.info("Got validation set")
        if tensorboard:
            tb = [TensorBoard(log_dir=self.output_dir)]
        else:
            tb = []

        # train the model
        if len(self.valid_dataset) == 0:
            raise ValueError("len(self.valid_dataset) == 0")

        logger.info("Started fitting generator")

        self.model.fit_generator(train_it,
                                 epochs=epochs,
                                 steps_per_epoch=max(int(len(self.train_dataset) / batch_size * train_epoch_frac), 1),
                                 validation_data=valid_it,
                                 validation_steps=max(int(len(self.valid_dataset) / batch_size * valid_epoch_frac), 1),
                                 callbacks=[EarlyStopping(patience=early_stop_patience),
                                            CSVLogger(self.history_path),
                                            ModelCheckpoint(self.ckp_file, save_best_only=True)] + tb
                                 )
        self.model = load_model(self.ckp_file)

    #     def load_best(self):
    #         """Load the best model from the Checkpoint file
    #         """
    #         self.model = load_model(self.ckp_file)

    def evaluate(self, metric, batch_size=256, num_workers=8, save=True):
        """Evaluate the model on the validation set
        Args:
          metrics: a list or a dictionary of metrics
          batch_size:
          num_workers:
        """
        lpreds = []
        llabels = []
        for inputs, targets in tqdm(self.valid_dataset.batch_train_iter(cycle=False,
                                                                        num_workers=num_workers,
                                                                        batch_size=batch_size),
                                    total=len(self.valid_dataset) // batch_size
                                    ):
            lpreds.append(self.model.predict_on_batch(inputs))
            llabels.append(targets)
        preds = numpy_collate_concat(lpreds)
        labels = numpy_collate_concat(llabels)
        del lpreds
        del llabels
        metric_res = metric(labels, preds)

        if save:
            write_json(metric_res, self.evaluation_path, indent=2)

        if self.cometml_experiment:
            self.cometml_experiment.log_multiple_metrics(flatten(metric_res), prefix="best/")

        return metric_res


@gin.configurable
class SklearnLogisticRegressionTrainer:
    """Simple Scikit model trainer
    """

    def __init__(self,
                 model,
                 train_dataset,
                 valid_dataset,
                 output_dir,
                 cometml_experiment=None):
        """
        Args:
          model: 
          train: training Dataset (object inheriting from kipoi.data.Dataset)
          valid: validation Dataset (object inheriting from kipoi.data.Dataset)
          output_dir: output directory where to log the training
          cometml_experiment: if not None, append logs to commetml
        """
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.cometml_experiment = cometml_experiment

        # setup the output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ckp_file = f"{self.output_dir}/model.h5"
        if os.path.exists(self.ckp_file):
            raise ValueError(f"model.h5 already exists in {self.output_dir}")
        self.history_path = f"{self.output_dir}/history.csv"
        self.evaluation_path = f"{self.output_dir}/evaluation.valid.json"


    def train(self,
              sample_weight=None,
              scaler_path=None,
              training_type=np.float32,
              **kwargs):
        
        # **kwargs won't be used, they are just included for compatibility with gin_train.
        """Train the model
        Args:
          batch_size:
          num_workers: how many workers to use in parallel
        """
        from sklearn.externals import joblib

        print("Started loading training dataset")
        
        X_train, y_train = self.train_dataset.load_all()

        if len(self.valid_dataset) == 0:
            raise ValueError("len(self.valid_dataset) == 0")

        if scaler_path:

            scaler = load_pickle(scaler_path)
            print("Started scaling X.")
            X_infl = X_train.astype(np.float32)
            X_infl = scaler.transform(X_infl)

            if training_type is not np.float32:    
                X_train = X_infl.astype(np.float16)

                from scipy.sparse import csr_matrix
                if isinstance(X_train, csr_matrix):
                    X_train.data = np.minimum(X_train.data, 65500)
                else:
                    X_train = np.minimum(X_train, 65500)
                del X_infl
                print("The dataset was downscaled.")
            print("Finished scaling X.")
        
        print("Finished loading training dataset. Shape: ", X_train.shape, "True values:", y_train.sum()/y_train.shape[0])
        self.model.fit(X_train,
                       y_train,
                       sample_weight=sample_weight)
        
        print("Calculating training accuracy:")
        acc = self.model.score(X_train, y_train)
        print("Obtained training accuracy: ", acc)

        joblib.dump(self.model, self.ckp_file)


    def evaluate(self,
                 eval_metric,
                 scaler_path=None,
                 eval_type=np.float32,
                 save=True,
                 **kwargs):
                
        # **kwargs won't be used, they are just included for compatibility with gin_train.
        """Evaluate the model on the validation set
        Args:
          metrics: a list or a dictionary of metrics
          batch_size:
          num_workers:
        """
        print("Started loading validation dataset")
        
        X_valid, y_valid = self.valid_dataset.load_all()

        if scaler_path:
            scaler = load_pickle(scaler_path)
            print("Started scaling X.")
            X_infl = X_valid.astype(np.float32)
            X_infl = scaler.transform(X_infl)

            if eval_type is not np.float32:
                X_valid = X_infl.astype(np.float16)

                from scipy.sparse import csr_matrix
                if isinstance(X_valid, csr_matrix):
                    X_valid.data = np.minimum(X_valid.data, 65500)
                else:
                    X_valid = np.minimum(X_valid, 65500)
                del X_infl
            print("Finished scaling X.")

        print("Finished loading validation dataset. Shape: ", X_valid.shape, "True values:", y_valid.sum()/y_valid.shape[0])
        
        y_pred = self.model.predict(X_valid)
        metric_res = eval_metric(y_valid, y_pred)
        print("metric_res", metric_res, np.amax(X_valid))

        if save:
            write_json(metric_res, self.evaluation_path, indent=2)

        if self.cometml_experiment:
            self.cometml_experiment.log_multiple_metrics(flatten(metric_res), prefix="best/")

        return metric_res