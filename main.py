import jax
import jax.numpy as jnp
from model import OptimizedCNN
from data import generate_synthetic_data, create_data_loader, evaluate
from train import train

def main():
    print("开始生成随机数种子...")
    rng = jax.random.PRNGKey(42)
    
    print("生成合成数据...")
    train_data = generate_synthetic_data()
    val_data = generate_synthetic_data()
    
    assert isinstance(train_data, tuple) and len(train_data) == 2, "train_data 必须是二元组"
    assert isinstance(val_data, tuple) and len(val_data) == 2, "val_data 必须是二元组"
    
    print("train_data type:", type(train_data))
    print("val_data type:", type(val_data))
    
    if isinstance(train_data[0], (list, tuple)):
        train_size = len(train_data[0])
    else:
        train_size = train_data[0].shape[0]
        
    if isinstance(val_data[0], (list, tuple)):
        val_size = len(val_data[0])
    else:
        val_size = val_data[0].shape[0]
        
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")
    
    print("开始训练 (50轮)...")
    trained_state = train(
        model=OptimizedCNN(num_classes=10),
        train_data=train_data,
        val_data=val_data,
        batch_size=32,
        epochs=50,
        print_interval=5
    )
    
    print("开始测试...")
    test_data = generate_synthetic_data(train=False)
    test_loader = create_data_loader(test_data, batch_size=32)
    accuracy = evaluate(trained_state, test_loader)
    print(f"最终测试准确率: {accuracy:.4f}")

if __name__ == "__main__":
    main()
