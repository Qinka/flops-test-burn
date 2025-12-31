// 参考 https://zhuanlan.zhihu.com/p/32479600270

use std::fmt::Display;

use burn::{Tensor, prelude::Backend};
use clap::{Parser, ValueEnum, command};
use tracing::{debug, info};

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum BackendType {
  /// with cpu
  #[cfg(feature = "cpu")]
  Cpu,
  /// with ndarray
  #[cfg(feature = "ndarray")]
  Ndarray,
  /// with wgpu
  #[cfg(feature = "wgpu")]
  Wgpu,
  /// with vulkan
  #[cfg(feature = "vulkan")]
  Vulkan,
  /// with cuda
  #[cfg(feature = "cuda")]
  Cuda,
}

impl Display for BackendType {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      #[cfg(feature = "cpu")]
      BackendType::Cpu => write!(f, "cpu"),
      #[cfg(feature = "ndarray")]
      BackendType::Ndarray => write!(f, "ndarray"),
      #[cfg(feature = "wgpu")]
      BackendType::Wgpu => write!(f, "wgpu"),
      #[cfg(feature = "vulkan")]
      BackendType::Vulkan => write!(f, "vulkan"),
      #[cfg(feature = "cuda")]
      BackendType::Cuda => write!(f, "cuda"),
    }
  }
}

#[derive(Parser, Debug)]
#[command(name = "mnist_burn", about = "MNIST  DeepLearning with Burn(Rust)")]
pub struct FlopsTest {
  #[arg(long, help = "size of the matrix", default_value_t = 1024)]
  pub matrix_size: usize,
  #[arg(long, help = "repeat times", default_value_t = 100)]
  pub repeat_times: usize,
  #[arg(long, help = "Backend to use")]
  pub backend: BackendType,
}

fn main() {
  tracing_subscriber::fmt::init();

  let args = FlopsTest::parse();
  debug!("Parsed arguments: {:?}", args);

  info!(
    "Running FLOPS test with matrix size {} and repeat times {} using backend {}",
    args.matrix_size, args.repeat_times, args.backend
  );

  match args.backend {
    #[cfg(feature = "cpu")]
    BackendType::Cpu => {
      info!("Using CPU backend for FLOPS test");
      let device = Default::default();
      flops_test::<burn::backend::Cpu>(&device, args.matrix_size, args.repeat_times)
    }
    #[cfg(feature = "ndarray")]
    BackendType::Ndarray => {
      info!("Using NDArray backend for FLOPS test");
      let device = Default::default();
      flops_test::<burn::backend::NdArray>(&device, args.matrix_size, args.repeat_times)
    }
    #[cfg(feature = "wgpu")]
    BackendType::Wgpu => {
      info!("Using WGPU backend for FLOPS test");
      let device = Default::default();
      flops_test::<burn::backend::Wgpu>(&device, args.matrix_size, args.repeat_times)
    }
    #[cfg(feature = "vulkan")]
    BackendType::Vulkan => {
      info!("Using Vulkan backend for FLOPS test");
      let device = Default::default();
      flops_test::<burn::backend::Vulkan>(&device, args.matrix_size, args.repeat_times)
    }
    #[cfg(feature = "cuda")]
    BackendType::Cuda => {
      info!("Using CUDA backend for FLOPS test");
      let device = Default::default();
      flops_test::<burn::backend::Cuda>(&device, args.matrix_size, args.repeat_times)
    }
  }
}

fn flops_test<B: Backend>(device: &B::Device, matrix_size: usize, repeat_times: usize) {
  let a = Tensor::<B, 2>::random([matrix_size, matrix_size], Default::default(), device);
  let b = Tensor::<B, 2>::random([matrix_size, matrix_size], Default::default(), device);

  let now = std::time::Instant::now();
  for _ in 0..repeat_times {
    let _c = Tensor::<B, 2>::matmul(a.clone(), b.clone());
  }
  let duration = now.elapsed();

  let flops = 2.0 * (matrix_size as f64).powi(3) * (repeat_times as f64);
  let gflops = flops / duration.as_secs_f64() / 1e9;

  println!(
    "Matrix size: {}x{}, Time: {:?}, GFLOPS: {:.2}",
    matrix_size, matrix_size, duration, gflops
  );
}
