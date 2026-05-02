import torch
import matplotlib.pyplot as plt
import logging
import traceback
from transfer_learning_fixed import ReferenceParserTrainer, MODEL_CONFIGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_metrics_fix")

def test_metric_calculation():
    """Test the fixed metric calculation function with both tensor and list inputs."""
    # Create a simple trainer instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = ReferenceParserTrainer(
        model_type="reference_parser",
        model_dir="test_model",
        device=device
    )
    
    # Test with tensor inputs
    logging.info("Testing with tensor inputs...")
    preds_tensor = torch.tensor([[0, 1, 2, 0, 1], [1, 2, 0, 1, 0]])
    labels_tensor = torch.tensor([[0, 1, 2, 0, -100], [1, 2, 0, -100, -100]])
    try:
        metrics = trainer.calculate_metrics(preds_tensor, labels_tensor)
        logging.info("Tensor test successful!")
        logging.info(f"Metrics: {metrics}")
    except Exception as e:
        logging.error(f"Tensor test failed: {e}")
        logging.error(traceback.format_exc())
    
    # Test with list inputs
    logging.info("\nTesting with list inputs...")
    preds_list = [[0, 1, 2, 0, 1], [1, 2, 0, 1, 0]]
    labels_list = [[0, 1, 2, 0, -100], [1, 2, 0, -100, -100]]
    try:
        metrics = trainer.calculate_metrics(preds_list, labels_list)
        logging.info("List test successful!")
        logging.info(f"Metrics: {metrics}")
    except Exception as e:
        logging.error(f"List test failed: {e}")
        logging.error(traceback.format_exc())
    
    # Test entity F1 calculation
    logging.info("\nTesting entity F1 calculation...")
    try:
        entity_f1 = trainer._calculate_entity_f1(preds_list, labels_list)
        logging.info("Entity F1 test successful!")
        logging.info(f"Entity F1: {entity_f1}")
    except Exception as e:
        logging.error(f"Entity F1 test failed: {e}")
        logging.error(traceback.format_exc())

def test_plotting():
    """Test the plotting function with various input scenarios."""
    # Create a simple trainer instance
    trainer = ReferenceParserTrainer(
        model_type="reference_parser",
        model_dir="test_model"
    )
    
    # Test case 1: Normal stats
    logging.info("\nTesting plotting with normal stats...")
    trainer.training_stats = {
        'train_loss': [0.5, 0.4, 0.3],
        'val_loss': [0.6, 0.5, 0.4],
        'val_metrics': [
            {'accuracy': 0.8, 'f1': 0.7},
            {'accuracy': 0.85, 'f1': 0.75},
            {'accuracy': 0.9, 'f1': 0.8}
        ],
        'learning_rates': [0.001, 0.0005, 0.0001]
    }
    try:
        trainer.plot_learning_curves()
        logging.info("Normal plotting test successful!")
    except Exception as e:
        logging.error(f"Normal plotting test failed: {e}")
        logging.error(traceback.format_exc())
    
    # Test case 2: Empty val_loss
    logging.info("\nTesting plotting with empty val_loss...")
    trainer.training_stats = {
        'train_loss': [0.5, 0.4, 0.3],
        'val_loss': [],
        'val_metrics': [
            {'accuracy': 0.8, 'f1': 0.7},
            {'accuracy': 0.85, 'f1': 0.75},
            {'accuracy': 0.9, 'f1': 0.8}
        ],
        'learning_rates': [0.001, 0.0005, 0.0001]
    }
    try:
        trainer.plot_learning_curves()
        logging.info("Empty val_loss test successful!")
    except Exception as e:
        logging.error(f"Empty val_loss test failed: {e}")
        logging.error(traceback.format_exc())
    
    # Test case 3: Different lengths
    logging.info("\nTesting plotting with different lengths...")
    trainer.training_stats = {
        'train_loss': [0.5, 0.4, 0.3],
        'val_loss': [0.6],
        'val_metrics': [
            {'accuracy': 0.8, 'f1': 0.7},
        ],
        'learning_rates': [0.001, 0.0005]
    }
    try:
        trainer.plot_learning_curves()
        logging.info("Different lengths test successful!")
    except Exception as e:
        logging.error(f"Different lengths test failed: {e}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    logging.info("Testing metric calculation functions...")
    test_metric_calculation()
    
    logging.info("\nTesting plotting functions...")
    test_plotting()
    
    logging.info("\nAll tests complete!") 