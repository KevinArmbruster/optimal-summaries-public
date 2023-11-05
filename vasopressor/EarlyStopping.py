from enum import Enum
import copy

class EarlyStopping:
    
    class Mode(Enum):
        MIN = 0
        MAX = 1
    
    def __init__(self, patience=3, min_delta=0, mode=Mode.MIN):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.mode = mode
        self.min_max_criterion = float('inf') if mode == EarlyStopping.Mode.MIN else float('-inf')
        self.best_state = 0

    def check_improvement(self, criterion, state):
        if self.mode == EarlyStopping.Mode.MIN:
            if criterion < (self.min_max_criterion - self.min_delta):
                self.min_max_criterion = criterion
                self.counter = 0
                self.best_state = copy.deepcopy(state)
            else:
                self.counter += 1
        
        elif self.mode == EarlyStopping.Mode.MAX:
            if criterion > (self.min_max_criterion + self.min_delta):
                self.min_max_criterion = criterion
                self.counter = 0
                self.best_state = copy.deepcopy(state)
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            print(f"Early Stopped")
            return True
        return False
    
    def has_stopped(self):
        if self.counter >= self.patience:
            return True
        return False
