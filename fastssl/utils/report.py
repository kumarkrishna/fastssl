"""
Generate a report from a number of checkpoints.

For each checkpoint, we will:
    1. Load the checkpoint
    2. Train linear classifier with pretrained backbone
    3. Report accuracy on test set
    4. Compute alpha on test set 
    5. generate plots 
        - accuracy vs epoch
        - alpha vs epoch
        - accuracy vs alpha
"""

class Report():
    super().__init__()
    def __init__(self, ckpt_dir, epoch_range=(1, 100), metrics=None):
        self.ckpt_dir = ckpt_dir
        self.epoch_begin, self.epoch_end = epoch_range
        self.metrics = metrics

    def get_checkpoints(self, args):
        """
        Get the checkpoints from the ckpt_dir.
        """
        checkpoints = []
        for ckpt in os.listdir(self.ckpt_dir):
            if ckpt.endswith('pth'):
                checkpoints.append(ckpt)
        return checkpoints

    def load_model():
        """
        Load the model.
        """
        pass

    def eval_classifier():
        """
        Evaluate the linear classifier.
        """
        pass


    def run(self, args):
        """
        Run the report.
        """
        # get the checkpoints
        checkpoints = self.get_checkpoints(args)
        # get the metrics
        metrics = self.get_metrics(args)
        # get the report
        report = self.get_report(args, checkpoints, metrics)
        return report