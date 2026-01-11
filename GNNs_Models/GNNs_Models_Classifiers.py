import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator

    
# GraphSAGE model: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GraphSAGE.html
class GraphSAGEClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GraphSAGEClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # Define layers
        self.conv1 = SAGEConv(in_channels=input_dim, out_channels=hidden_dim)
        self.conv2 = SAGEConv(in_channels=hidden_dim, out_channels=output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x




class MyGNNClassifier(BaseEstimator, ClassifierMixin, nn.Module):
    def __init__(self, metric='accuracy', verbose=False, model_name='GraphSAGE'):
        super(MyGNNClassifier, self).__init__()
        # User-facing attributes
        self.metric = metric
        self.verbose = verbose
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Hyperparameters (to be set via set_params / GridSearch)
        self.input_dim = None
        self.hidden_dim = None
        self.num_heads = None
        self.output_dim = None
        self.dropout = None
        self.learning_rate = None
        self.weight_decay = None
        self.batch_size = None
        self.epochs = None

        # runtime objects
        self.model = None
        self.optimizer = None
        self.criterion = None

    # -------------------------
    # sklearn-compatible params
    # -------------------------
    def set_params(self, **params):
        """
        Store parameters on the estimator object. Do NOT create the model here.
        GridSearchCV will call set_params repeatedly; model creation should happen in fit().
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # -------------------------
    # internal model init
    # -------------------------
    def _init_model_and_optim(self):
        # sensible defaults if not provided
        if self.hidden_dim is None:
            self.hidden_dim = 32
        if self.num_heads is None:
            self.num_heads = 1
        if self.output_dim is None:
            self.output_dim = 2
        if self.dropout is None:
            self.dropout = 0.0
        if self.learning_rate is None:
            self.learning_rate = 1e-2
        if self.weight_decay is None:
            self.weight_decay = 0.0
        if self.batch_size is None:
            self.batch_size = 32
        if self.epochs is None:
            self.epochs = 10
        if self.input_dim is None:
            raise RuntimeError("input_dim must be set (either via set_params or inferred before calling _init_model_and_optim).")

        # instantiate model according to model_name
        if self.model_name == 'GraphSAGE':
            self.model = GraphSAGEClassifier(input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                                     output_dim=self.output_dim, dropout=self.dropout).to(self.device)
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")

        # optimizer & default loss (may be overridden by class weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

    # -------------------------
    # training helper
    # -------------------------
    def _training_epoch(self, train_loader):
        """
        Train one epoch on DataLoader and return summed loss (float).
        """
        self.model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = batch.to(self.device)
            # ensure labels are shape [batch] and long
            y = batch.y.view(-1).long().to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(batch)   # expect shape [batch_size, num_classes]
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss

    # -------------------------
    # fit
    # -------------------------
    def fit(self, train_data, y=None):
        """
        Fit the GNN using PyG DataLoader, with sklearn-compatible signature.
        Logs loss, recall, F1 per epoch, and handles class imbalance.
        """
        # If y provided, sanity-check lengths
        if y is not None:
            try:
                if len(y) != len(train_data):
                    print("Warning: y length does not match train_data length; proceeding using labels from train_data.")
            except Exception:
                pass

        # compute class counts
        cnt = Counter()
        for d in train_data:
            y_ = d.y
            if isinstance(y_, torch.Tensor) and y_.numel() == 1:
                cnt[int(y_.item())] += 1
            else:
                cnt[int(y_.view(-1).argmax().item())] += 1
        pos = cnt.get(1, 0)
        neg = cnt.get(0, 0)

        # class weights
        if pos > 0 and neg > 0:
            #weights = torch.tensor([neg/(pos+neg), pos/(pos+neg)], dtype=torch.float, device=self.device)
            #self.criterion = nn.CrossEntropyLoss(weight=weights)
            self.criterion = nn.CrossEntropyLoss()

        else:
            self.criterion = nn.CrossEntropyLoss()

        # initialize model & optimizer
        self._init_model_and_optim()

        # Weighted sampler for balanced batches
        sample_weights = []
        for d in train_data:
            y_val = int(d.y.view(-1).item()) if isinstance(d.y, torch.Tensor) else int(d.y)
            sample_weights.append(1.0 / (pos if y_val == 1 else neg))

        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(train_data), replacement=True)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, sampler=sampler)

        train_losses = []
        for epoch in range(self.epochs):
            running_loss = 0.0
            self.model.train()
            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                y_batch = batch.y.view(-1).long().to(self.device)
                output = self.model(batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / max(1, len(train_loader))
            train_losses.append(epoch_loss)

            # log metrics per epoch
            #acc, prec, rec, f1, auc = self.predict_metrics(train_data)
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.epochs} — Loss: {epoch_loss:.4f}")

        # Required for sklearn compatibility
        self.classes_ = np.array([0, 1])

        return self
    

    # -------------------------
    # predict / predict_proba
    # -------------------------
    def predict_proba(self, test_data):
        """
        Returns an (n_samples, n_classes) numpy array of probabilities (float).
        Also writes logs via log_results_to_csv if you want batch logging.
        """
        self.model.eval()
        loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        probs_list = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch)               # [batch, C]
                probs = F.softmax(logits, dim=1).cpu().numpy()  # numpy array
                probs_list.append(probs)

                # optional logging per-batch (if you want)
                # For compatibility with your old log function expect predictions and pos probs
                try:
                    preds = np.argmax(probs, axis=1)
                    pos_probs = probs[:, 1]
                    self.log_results_to_csv(batch, torch.from_numpy(preds), pos_probs, tot_len=len(test_data))
                except Exception:
                    # don't fail logging if shapes differ; logging is optional
                    pass

        return np.vstack(probs_list) if len(probs_list) > 0 else np.empty((0, self.output_dim))

    def predict(self, test_data):
        """
        Returns 1-D numpy array of predicted class labels (ints).
        Compatible with sklearn scorers.
        """
        probs = self.predict_proba(test_data)
        if probs.size == 0:
            return np.array([], dtype=int)
        preds = probs.argmax(axis=1)
        return preds.astype(int)

    # -------------------------
    # convenience evaluation
    # -------------------------
    def predict_metrics(self, test_data):
        """
        Compute accuracy, precision, recall, f1, auc on test_data and return tuple.
        Useful for logging and debugging.
        """
        probs = self.predict_proba(test_data)
        if probs.size == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        preds = probs.argmax(axis=1)
        pos_probs = probs[:, 1]

        # retrieve true labels
        trues = []
        loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        for batch in loader:
            y_batch = batch.y.view(-1).long().cpu().numpy()
            trues.append(y_batch)
        trues = np.concatenate(trues)

        accuracy = (preds == trues).sum() / max(1, trues.shape[0])
        precision = precision_score(trues, preds, zero_division=0)
        recall = recall_score(trues, preds, zero_division=0)
        f1 = f1_score(trues, preds, zero_division=0)
        try:
            auc = roc_auc_score(trues, pos_probs)
        except ValueError:
            auc = 0.0
        return accuracy, precision, recall, f1, auc

    # -------------------------
    # logging helper (kept from your implementation)
    # -------------------------
    def log_results_to_csv(self, data, predictions, probabilities, tot_len):
        csv_file = "analyzed_validation_results.csv"
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "filename", 
                    "name", "address", "node_shape", "edge_shape",
                    "true_label", "prediction", "parser_probability", "dataset_len",
                    "hidden_dim", "dropout", "learning_rate",
                    "epochs", "batch_size", "weight_decay", "num_heads"
                ])

            for i in range(data.num_graphs):
                # pick the filename for the i-th graph
                file_i = data.filename[i]
                writer.writerow([
                    file_i,  
                    data.ref.get('name', ['N/A'])[i],
                    data.ref.get('address', ['N/A'])[i],
                    tuple(data.x[data.batch == i].shape),
                    tuple(data.edge_index[:, data.batch[data.edge_index[0]] == i].shape),
                    int(data.y[i].item()),
                    int(predictions[i].item()) if hasattr(predictions, "__getitem__") else int(predictions.item()),
                    float(probabilities[i]) if isinstance(probabilities, (list, np.ndarray)) else float(probabilities),
                    tot_len,
                    self.hidden_dim,
                    self.dropout,
                    self.learning_rate,
                    self.epochs,
                    self.batch_size,
                    self.weight_decay,
                    self.num_heads
                ])

    def predict_case_studies(self, test_data):
        """
        Return per-graph predictions in a DataFrame.
        """
        self.model.eval()
        test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        results = []

        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
                predicted = np.argmax(probabilities)

                name = data.ref['name'][0] if 'name' in data.ref else 'N/A'
                address = data.ref['address'][0] if 'address' in data.ref else 'N/A'

                results.append({
                    'Name': name,
                    'Address': address,
                    'Predicted': int(predicted),
                    'Probability_0': float(probabilities[0]),
                    'Probability_1': float(probabilities[1]),
                    'Num_Nodes': data.num_nodes,
                    'Num_Edges': data.num_edges
                })

        return pd.DataFrame(results)

    # -------------------------
    # score (sklearn compatibility)
    # -------------------------
    def score(self, test_data, case_studies=False, filename=None):
        """
        Evaluate the model on test_data.

        Parameters:
            test_data : list of PyG Data objects
            case_studies : bool, if True, save per-graph predictions to CSV
            filename : str or None, CSV path for case studies

        Returns:
            If case_studies=False: metric value (accuracy/precision/recall/f1)
            If case_studies=True: pandas DataFrame with predictions
        """
        if not case_studies:
            # Standard evaluation
            accuracy, precision, recall, f1, auc_roc = self.predict(test_data)

            metric_value = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }.get(self.metric, accuracy)

            print(f"[Score] Metric ({self.metric}): {metric_value:.4f}")
            return metric_value
        else:
            # Case studies evaluation
            df = self.predict_case_studies(test_data)
            # Round probabilities
            if "Probability_0" in df.columns and "Probability_1" in df.columns:
                df["Probability_0"] = df["Probability_0"].round(3)
                df["Probability_1"] = df["Probability_1"].round(3)
            # Save to CSV if requested
            if filename:
                print(f"[Score] Saving case studies to CSV: {filename}")
                df.to_csv(filename, index=False)
            return df