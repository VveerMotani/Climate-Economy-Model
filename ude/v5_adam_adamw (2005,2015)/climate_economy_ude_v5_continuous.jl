#=
================================================================================
Climate-Economy UDE Model - VERSION 5 CONTINUOUS: ADAM + ADAMW (NO BFGS)
================================================================================
This script treats data as CONTINUOUS using interpolation rather than discrete
points. The model uses:
  - DataInterpolations.jl to create smooth interpolants of observed data
  - Continuous loss computed over the interpolated trajectory
  - Integration-based error metric instead of point-wise MSE

Optimization Strategy:
  Stage 1: ADAM (exploration, lr=0.005)
  Stage 2: ADAMW (weight decay, lr=0.002)
  Stage 3: ADAM (fine-tuning, lr=0.0005)

Train/Test Split: 1959-2005 (train) | 2006-2023 (test)
================================================================================
=#

# ==============================================================================
# 1. PACKAGE IMPORTS
# ==============================================================================
using DifferentialEquations
using SciMLSensitivity
using Lux
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using DataFrames
using CSV
using Plots
using ComponentArrays
using Random
using Statistics
using DataInterpolations  # For continuous interpolation

println("✓ All packages loaded successfully!")

# ==============================================================================
# 2. DATA PREPARATION
# ==============================================================================
println("\n" * "="^60)
println("Loading and preparing data (CONTINUOUS APPROACH)...")
println("="^60)

# Load the CSV file
data_path = joinpath(@__DIR__, "..", "..", "..", "data", "final_training_data_corrected_temp.csv")
df = CSV.read(data_path, DataFrame)

# Reference values for un-normalization (1959 baseline values)
const K_1959 = 44447037.359436  # Capital stock in mil. 2021US$
const Y_1959 = 12934161.3005371  # Real GDP in mil. 2021US$
const C_1959 = 315.98  # CO2 concentration in ppm
const T_1959 = 0       # Temperature anomaly in 1959 (°C)

# Extract time vector (normalized to start at 0)
years_all = df.Year
t_data_all = Float64.(years_all .- years_all[1])  # Time from 0 to 64 (1959-2023)

# Extract state variables as column vectors (FULL dataset)
Y_data_all = Float64.(df.Y_norm)      # GDP (normalized)
T_data_all = Float64.(df.T_adjusted)  # Temperature (adjusted)
C_data_all = Float64.(df.C_norm)      # CO2 (normalized)
K_data_all = Float64.(df.K_norm)      # Capital (normalized)

# ==============================================================================
# TRAIN-TEST SPLIT: Train on 1959-2005, Test/Predict on 2006-2023
# ==============================================================================
train_end_year = 2015
test_end_year = 2023

# Find indices for train and test periods
train_idx = findall(y -> y <= train_end_year, years_all)
test_idx = findall(y -> y > train_end_year && y <= test_end_year, years_all)
full_idx = findall(y -> y <= test_end_year, years_all)

# Training data (1959-2005)
years_train = years_all[train_idx]
t_train = t_data_all[train_idx]
Y_train = Y_data_all[train_idx]
T_train = T_data_all[train_idx]
C_train = C_data_all[train_idx]
K_train = K_data_all[train_idx]

# Test data (2006-2023)
years_test = years_all[test_idx]
t_test = t_data_all[test_idx]
Y_test = Y_data_all[test_idx]
T_test = T_data_all[test_idx]
C_test = C_data_all[test_idx]
K_test = K_data_all[test_idx]

# Full period for prediction (1959-2023)
years = years_all[full_idx]
t_data = t_data_all[full_idx]
Y_data = Y_data_all[full_idx]
T_data = T_data_all[full_idx]
C_data = C_data_all[full_idx]
K_data = K_data_all[full_idx]

# ==============================================================================
# 3. CREATE CONTINUOUS INTERPOLANTS OF TRAINING DATA
# ==============================================================================
println("\n" * "="^60)
println("Creating CONTINUOUS interpolants of training data...")
println("="^60)

# Use CubicSpline interpolation for smooth continuous representation
# This treats the discrete data as samples from a continuous process
Y_interp = CubicSpline(Y_train, t_train)
T_interp = CubicSpline(T_train, t_train)
C_interp = CubicSpline(C_train, t_train)
K_interp = CubicSpline(K_train, t_train)

# Also create full data interpolants for visualization
Y_interp_full = CubicSpline(Y_data, t_data)
T_interp_full = CubicSpline(T_data, t_data)
C_interp_full = CubicSpline(C_data, t_data)
K_interp_full = CubicSpline(K_data, t_data)

println("✓ Created CubicSpline interpolants for all state variables")
println("  Interpolation type: CubicSpline (C2 continuous)")

# Training data matrix (for comparison - still used for discrete loss reference)
train_matrix = vcat(Y_train', T_train', C_train', K_train')
n_train = length(train_idx)

# Full data matrix (for visualization)
data_matrix = vcat(Y_data', T_data', C_data', K_data')
n_states, n_times = size(data_matrix)

# Time spans
tspan_train = (t_train[1], t_train[end])  # Training: 1959-2005
tspan = (t_data[1], t_data[end])            # Full prediction: 1959-2023

# Initial conditions from data
u0 = train_matrix[:, 1]

# Reference C₀ for the log term (first value)
C0_ref = C_train[1]

println("\n" * "="^60)
println("TRAIN-TEST SPLIT CONFIGURATION (CONTINUOUS)")
println("="^60)
println("  Training period: $(years_train[1]) to $(years_train[end]) ($(n_train) discrete points)")
println("  Test period:     $(years_test[1]) to $(years_test[end]) ($(length(test_idx)) discrete points)")
println("  Prediction span: $(years[1]) to $(years[end]) ($(n_times) discrete points)")
println("  *** Data is treated as CONTINUOUS via interpolation ***")

println("\nTraining Data Statistics:")
println("  Y: min=$(round(minimum(Y_train), digits=3)), max=$(round(maximum(Y_train), digits=3)), range=$(round(maximum(Y_train)-minimum(Y_train), digits=3))")
println("  T: min=$(round(minimum(T_train), digits=3)), max=$(round(maximum(T_train), digits=3)), range=$(round(maximum(T_train)-minimum(T_train), digits=3))")
println("  C: min=$(round(minimum(C_train), digits=3)), max=$(round(maximum(C_train), digits=3)), range=$(round(maximum(C_train)-minimum(C_train), digits=3))")
println("  K: min=$(round(minimum(K_train), digits=3)), max=$(round(maximum(K_train), digits=3)), range=$(round(maximum(K_train)-minimum(K_train), digits=3))")

# Compute approximate derivatives from training data for parameter estimation
dY_approx = (Y_train[end] - Y_train[1]) / (t_train[end] - t_train[1])
dT_approx = (T_train[end] - T_train[1]) / (t_train[end] - t_train[1])
dC_approx = (C_train[end] - C_train[1]) / (t_train[end] - t_train[1])
dK_approx = (K_train[end] - K_train[1]) / (t_train[end] - t_train[1])

println("\nApproximate rates (from training data):")
println("  dY/dt ≈ $(round(dY_approx, digits=4))")
println("  dT/dt ≈ $(round(dT_approx, digits=4))")
println("  dC/dt ≈ $(round(dC_approx, digits=4))")
println("  dK/dt ≈ $(round(dK_approx, digits=4))")

println("\n✓ Data loaded and interpolated successfully!")
println("  Initial conditions: Y₀=$(u0[1]), T₀=$(u0[2]), C₀=$(u0[3]), K₀=$(u0[4])")

# ==============================================================================
# 4. NEURAL NETWORK DEFINITION (Lux.jl)
# ==============================================================================
println("\n" * "="^60)
println("Defining Neural Network architecture...")
println("="^60)

# Set random seed for reproducibility
rng = Random.default_rng()
Random.seed!(rng, 42)

# Define the neural network: C(t) → learned forcing term
# Architecture: Input(1) → Dense(32, tanh) → Dense(32, tanh) → Dense(1)
# Added output scaling layer to match expected forcing magnitude (~0.01-0.02 scale)
nn_forcing = Lux.Chain(
    Lux.Dense(1 => 32, tanh),
    Lux.Dense(32 => 32, tanh),
    Lux.Dense(32 => 1),
    Lux.WrappedFunction(x -> x .* 0.02f0)  # Scale output to appropriate magnitude
)

# Initialize NN parameters and state
ps_nn, st_nn = Lux.setup(rng, nn_forcing)
ps_nn = ComponentArray(ps_nn)

println("✓ Neural Network initialized")
println("  Architecture: 1 → 32 → 32 → 1 (tanh) → *0.02 scaling")
println("  Number of NN parameters: $(length(ps_nn))")

# ==============================================================================
# 5. PHYSICAL PARAMETER INITIALIZATION (DATA-INFORMED)
# ==============================================================================
println("\n" * "="^60)
println("Initializing physical parameters (data-informed)...")
println("="^60)

#=
Physical Parameters for the Climate-Economy ODE System:

GDP Equation: dY/dt = β*Y - ψ*T²*Y
  β (beta)  : Economic growth rate (baseline growth)
  ψ (psi)   : Temperature damage coefficient

CO2 Equation: dC/dt = θ*Y - ξ*(C - C₀)
  θ (theta) : Emissions intensity (CO2 per unit GDP)
  ξ (xi)    : Natural CO2 absorption rate

Temperature Equation: dT/dt = NN(C) - decay_param*T
  decay_param : Combined λ*κ term (radiative decay)
  NN(C)       : Learned forcing term (replaces λ*η*ln(C/C₀))

Capital Equation: dK/dt = s*Y - δ*K
  s (s)     : Savings/investment rate
  δ (delta) : Capital depreciation rate
=#

# Data-informed parameter initialization using TRAINING data
# Key insight: We need to match the observed growth patterns in the training period

# Compute average values from TRAINING data only
avg_Y_train = mean(Y_train)
avg_K_train = mean(K_train)
avg_C_train = mean(C_train)

# Estimate beta from GDP growth rate in training period
beta_estimate = log(Y_train[end] / Y_train[1]) / (t_train[end] - t_train[1])
beta_init = max(0.02, min(0.08, beta_estimate))  # Bounded to realistic range

# Estimate theta from CO2 dynamics: dC/dt ≈ θ*Y
theta_init = max(0.0001, dC_approx / avg_Y_train)

# Natural CO2 absorption rate (small)
xi_init = 0.005

# Estimate s and delta from capital dynamics  
delta_init = 0.04  # Typical depreciation ~4%
s_estimate = (dK_approx + delta_init * avg_K_train) / avg_Y_train
s_init = max(0.15, min(0.35, s_estimate))  # Bounded savings rate

# Temperature decay parameter - should be small for realistic climate dynamics
decay_init = 0.02

# Initialize physical parameters with better estimates
physical_params = ComponentArray(
    beta=beta_init,                  # GDP growth rate (~4-6%)
    psi=0.00001,                     # Temperature damage (start very small)
    theta=theta_init,                # Emissions intensity
    xi=xi_init,                      # CO2 absorption rate 
    s=s_init,                        # Savings rate (~25%)
    delta=delta_init,                # Depreciation rate (~4%)
    decay_param=decay_init           # Temperature decay
)

println("\nImproved parameter initialization (from training data):")
println("  beta (GDP growth rate): $(round(beta_init, digits=4)) ~ $(round(beta_init*100, digits=1))% per year")
println("  psi (temp damage):      $(physical_params.psi)")
println("  theta (emissions):      $(round(theta_init, digits=6))")
println("  xi (CO2 absorption):    $(round(xi_init, digits=4))")
println("  s (savings rate):       $(round(s_init, digits=4)) ~ $(round(s_init*100, digits=1))%")
println("  delta (depreciation):   $(round(delta_init, digits=4)) ~ $(round(delta_init*100, digits=1))%")
println("  decay_param:            $(round(decay_init, digits=4))")

# ==============================================================================
# 6. COMBINED PARAMETER VECTOR
# ==============================================================================
# Combine NN weights and physical parameters into single ComponentArray
p_init = ComponentArray(
    nn=ps_nn,
    phys=physical_params
)

println("\n✓ Combined parameter vector created")
println("  Total trainable parameters: $(length(p_init))")
println("    - Neural Network: $(length(ps_nn))")
println("    - Physical: $(length(physical_params))")

# ==============================================================================
# 7. HYBRID ODE SYSTEM DEFINITION
# ==============================================================================
println("\n" * "="^60)
println("Defining hybrid ODE system...")
println("="^60)

function climate_economy_ude!(du, u, p, t)
    # Unpack state variables
    Y, T, C, K = u

    # Unpack physical parameters
    β = p.phys.beta
    ψ = p.phys.psi
    θ = p.phys.theta
    ξ = p.phys.xi
    s = p.phys.s
    δ = p.phys.delta
    decay_param = p.phys.decay_param

    # Neural network forward pass: C → forcing term
    # NN replaces the theoretical λ*η*ln(C/C₀) term
    C_input = reshape([C], 1, 1)  # Shape: (1, 1) for Lux
    nn_out, _ = nn_forcing(C_input, p.nn, st_nn)
    forcing = nn_out[1]  # Extract scalar

    # === GDP Equation ===
    # dY/dt = β*Y - ψ*T²*Y
    du[1] = β * Y - ψ * T^2 * Y

    # === Temperature Equation (HYBRID) ===
    # dT/dt = NN(C) - decay_param*T
    # Original: dT/dt = λ*[η*ln(C/C₀) - κ*T]
    # NN learns: λ*η*ln(C/C₀) portion
    du[2] = (forcing - decay_param * T) / C0_ref

    # === CO2 Equation ===
    # dC/dt = θ*Y - ξ*(C - C₀)
    du[3] = θ * Y - ξ * (C - C0_ref)

    # === Capital Equation ===
    # dK/dt = s*Y - δ*K
    du[4] = s * Y - δ * K

    return nothing
end

println("✓ Hybrid ODE system defined")
println("  State variables: [Y, T, C, K]")
println("  Neural network: Active in dT/dt equation")

# ==============================================================================
# 8. ODE PROBLEM SETUP
# ==============================================================================
println("\n" * "="^60)
println("Setting up ODE problem...")
println("="^60)

# Create ODE problem for TRAINING (1959-2005)
prob_ude = ODEProblem(climate_economy_ude!, u0, tspan_train, p_init)

# Create ODE problem for FULL PREDICTION (1959-2023)
prob_ude_full = ODEProblem(climate_economy_ude!, u0, tspan, p_init)

# Test that the ODE runs with initial parameters
println("Testing ODE solver with initial parameters...")
try
    sol_test = solve(prob_ude, Tsit5(), saveat=t_train)
    println("✓ ODE solver test passed (training period)!")
    println("  Solution shape: $(size(Array(sol_test)))")
catch e
    println("✗ ODE solver test failed: $e")
end

# ==============================================================================
# 9. CONTINUOUS LOSS FUNCTION
# ==============================================================================
println("\n" * "="^60)
println("Defining CONTINUOUS loss function with interpolation...")
println("="^60)

# Storage for loss history
loss_history = Float64[]

# Number of integration points for continuous loss (denser than discrete data)
n_integration_points = 200  # Evaluate at 200 points over training period
t_integration = range(t_train[1], t_train[end], length=n_integration_points)

# Precompute interpolated training data at integration points
Y_target = [Y_interp(t) for t in t_integration]
T_target = [T_interp(t) for t in t_integration]
C_target = [C_interp(t) for t in t_integration]
K_target = [K_interp(t) for t in t_integration]

function loss_function_continuous(p, _)
    # Solve ODE with current parameters (TRAINING PERIOD)
    # Use dense output for continuous comparison
    prob = remake(prob_ude, p=p)
    sol = solve(prob, Tsit5(),
        saveat=collect(t_integration),  # Save at integration points
        sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true)),
        maxiters=1e6)

    # Check if solution was successful
    if sol.retcode != :Success && sol.retcode != ReturnCode.Success
        return Inf
    end

    # Get solution at integration points
    pred = Array(sol)  # 4 × n_integration_points

    # Compute CONTINUOUS L2 error (approximated by trapezoidal integration)
    # This is ∫(pred(t) - data(t))² dt / (T_end - T_start)

    dt = (t_train[end] - t_train[1]) / (n_integration_points - 1)

    # Squared errors at each integration point
    err_Y = (pred[1, :] .- Y_target) .^ 2
    err_T = (pred[2, :] .- T_target) .^ 2
    err_C = (pred[3, :] .- C_target) .^ 2
    err_K = (pred[4, :] .- K_target) .^ 2

    # Trapezoidal integration for each variable
    integral_Y = dt * (0.5 * err_Y[1] + sum(err_Y[2:end-1]) + 0.5 * err_Y[end])
    integral_T = dt * (0.5 * err_T[1] + sum(err_T[2:end-1]) + 0.5 * err_T[end])
    integral_C = dt * (0.5 * err_C[1] + sum(err_C[2:end-1]) + 0.5 * err_C[end])
    integral_K = dt * (0.5 * err_K[1] + sum(err_K[2:end-1]) + 0.5 * err_K[end])

    # Normalize by time interval to get mean integrated squared error
    time_span = t_train[end] - t_train[1]
    mse_continuous = (integral_Y + integral_T + integral_C + integral_K) / (4.0 * time_span)

    return mse_continuous
end

# Callback function for monitoring training progress
iter_count = Ref(0)

function callback(state, loss_val)
    iter_count[] += 1
    push!(loss_history, loss_val)

    # Report every 500 iterations (adjusted for longer training)
    if iter_count[] % 500 == 0 || iter_count[] == 1
        println("  Iteration $(iter_count[]): Loss = $(round(loss_val, digits=8))")
    end

    return false  # Continue optimization
end

# Test loss function
println("Testing CONTINUOUS loss function...")
initial_loss = loss_function_continuous(p_init, nothing)
println("✓ Initial loss (continuous): $(round(initial_loss, digits=6))")
println("  Integration points: $(n_integration_points)")
println("  Time resolution: $(round((t_train[end] - t_train[1]) / n_integration_points, digits=3)) years")

# ==============================================================================
# 10. TRAINING STAGE 1: ADAM OPTIMIZER (EXPLORATION)
# ==============================================================================
println("\n" * "="^60)
println("STAGE 1: ADAM optimizer (exploration)...")
println("="^60)

opt_func = OptimizationFunction(loss_function_continuous, Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, p_init)

adam_lr_1 = 0.005
adam_iters_1 = 5000

println("Settings: ADAM lr=$(adam_lr_1), iterations=$(adam_iters_1)")
println("Starting ADAM training (CONTINUOUS loss)...\n")

iter_count[] = 0

result_adam = solve(opt_prob, OptimizationOptimisers.Adam(adam_lr_1),
    maxiters=adam_iters_1,
    callback=callback)

p_adam = result_adam.u
loss_adam = result_adam.objective

println("\n✓ ADAM Stage complete! Loss: $(round(loss_adam, digits=6))")

# ==============================================================================
# 11. TRAINING STAGE 2: ADAMW (WEIGHT DECAY REGULARIZATION)
# ==============================================================================
println("\n" * "="^60)
println("STAGE 2: ADAMW optimizer (weight decay)...")
println("="^60)

opt_prob_2 = OptimizationProblem(opt_func, p_adam)

adamw_lr = 0.002
adamw_iters = 5000

println("Settings: ADAMW lr=$(adamw_lr), iterations=$(adamw_iters)")
println("Starting ADAMW training (CONTINUOUS loss)...\n")

result_adamw = solve(opt_prob_2, OptimizationOptimisers.AdamW(adamw_lr),
    maxiters=adamw_iters,
    callback=callback)

p_adamw = result_adamw.u
loss_adamw = result_adamw.objective

println("\n✓ ADAMW Stage complete! Loss: $(round(loss_adamw, digits=6))")

# ==============================================================================
# 12. TRAINING STAGE 3: ADAM FINE-TUNING (LOW LR)
# ==============================================================================
println("\n" * "="^60)
println("STAGE 3: ADAM fine-tuning (low lr)...")
println("="^60)

opt_prob_3 = OptimizationProblem(opt_func, p_adamw)

adam_lr_3 = 0.0005
adam_iters_3 = 4000

println("Settings: ADAM lr=$(adam_lr_3), iterations=$(adam_iters_3)")
println("Starting ADAM fine-tuning (CONTINUOUS loss)...\n")

result_final = solve(opt_prob_3, OptimizationOptimisers.Adam(adam_lr_3),
    maxiters=adam_iters_3,
    callback=callback)

p_final = result_final.u
loss_final = result_final.objective

println("\n✓ Training complete!")
println("  Final loss (continuous): $(round(loss_final, digits=6))")
println("  Total iterations: $(iter_count[])")

# Variables for plot labels
adam_iters_2 = adamw_iters
adam_lr_2 = adamw_lr

# ==============================================================================
# 13. FINAL SOLUTION AND PREDICTION
# ==============================================================================
println("\n" * "="^60)
println("Computing final solution and predictions...")
println("="^60)

# Solve ODE with optimized parameters on FULL PERIOD (1959-2023) for prediction
prob_final = remake(prob_ude_full, p=p_final)
sol_final = solve(prob_final, Tsit5(), saveat=t_data)
pred_final = Array(sol_final)

println("✓ Final solution computed (1959-2023)")
println("\nOptimized Physical Parameters:")
param_names = ["beta", "psi", "theta", "xi", "s", "delta", "decay_param"]
for (i, name) in enumerate(param_names)
    val = p_final.phys[i]
    println("  $name = $(round(val, digits=6))")
end

# Extract predictions for train and test periods
pred_train = pred_final[:, 1:n_train]  # Training period predictions
pred_test = pred_final[:, (n_train+1):end]  # Test period predictions

# Compute per-variable MSE for TRAINING period (DISCRETE comparison for reference)
println("\n" * "-"^40)
println("TRAINING PERIOD MSE (1959-$(train_end_year)) - Discrete Points:")
println("-"^40)
for (i, name) in enumerate(["Y (GDP)", "T (Temp)", "C (CO2)", "K (Capital)"])
    mse_i = mean((pred_train[i, :] .- train_matrix[i, :]) .^ 2)
    println("  $name: $(round(mse_i, digits=6))")
end

# Compute per-variable MSE for TEST period
test_matrix = vcat(Y_test', T_test', C_test', K_test')
println("\n" * "-"^40)
println("TEST PERIOD MSE ($(train_end_year+1)-$(test_end_year)) - OUT OF SAMPLE:")
println("-"^40)
for (i, name) in enumerate(["Y (GDP)", "T (Temp)", "C (CO2)", "K (Capital)"])
    mse_i = mean((pred_test[i, :] .- test_matrix[i, :]) .^ 2)
    println("  $name: $(round(mse_i, digits=6))")
end

# ==============================================================================
# 14. VISUALIZATION
# ==============================================================================
println("\n" * "="^60)
println("Generating visualizations (CONTINUOUS model)...")
println("="^60)

# --- Plot 1: Model vs Data (4 subplots) with UN-NORMALIZED VALUES ---
# Showing TRAIN (blue) vs TEST (green) data with model predictions
println("Creating Model vs Data plot with train/test split...")

# Convert normalized values back to original units for display
# Training data
Y_train_original = Y_train .* Y_1959 ./ 1e6  # Convert to Trillion 2021US$
K_train_original = K_train .* K_1959 ./ 1e6
C_train_original = C_train .* C_1959  # ppm
T_train_original = T_train .+ T_1959  # °C

# Test data
Y_test_original = Y_test .* Y_1959 ./ 1e6
K_test_original = K_test .* K_1959 ./ 1e6
C_test_original = C_test .* C_1959
T_test_original = T_test .+ T_1959

# Predictions (full period)
Y_pred_original = pred_final[1, :] .* Y_1959 ./ 1e6
K_pred_original = pred_final[4, :] .* K_1959 ./ 1e6
C_pred_original = pred_final[3, :] .* C_1959
T_pred_original = pred_final[2, :] .+ T_1959

# Optimization strategy info for plot title
opt_strategy = "ADAM+ADAMW+ADAM"
total_iters = adam_iters_1 + adamw_iters + adam_iters_3
lr_info = "lr: $(adam_lr_1)/$(adamw_lr)/$(adam_lr_3)"
split_info = "Train: 1959-$(train_end_year) | Test: $(train_end_year+1)-$(test_end_year)"

# Create main title with all info
main_title = "Climate-Economy UDE Model (CONTINUOUS)\\n" *
             "Strategy: $(opt_strategy) | Iters: $(total_iters) | $(lr_info)\\n" *
             "$(split_info) | Final Loss: $(round(loss_final, digits=6))"

p1 = plot(layout=(2, 2), size=(1600, 1100),
    plot_title=main_title,
    left_margin=15Plots.mm,   # Increased margin to prevent y-label cutoff
    bottom_margin=10Plots.mm,
    top_margin=5Plots.mm)

# GDP (Y) - Real GDP in Trillion 2021US$
plot!(p1[1], years_train, Y_train_original,
    seriestype=:scatter, label="Train (1959-$(train_end_year))",
    markersize=5, alpha=0.8, color=:blue)
plot!(p1[1], years_test, Y_test_original,
    seriestype=:scatter, label="Test ($(train_end_year+1)-$(test_end_year))",
    markersize=5, alpha=0.8, color=:green, markershape=:diamond)
plot!(p1[1], years, Y_pred_original,
    label="UDE Model (Continuous)", linewidth=2.5, color=:red)
vline!(p1[1], [train_end_year], linestyle=:dash, color=:gray, linewidth=2, label="")
xlabel!(p1[1], "Year")
ylabel!(p1[1], "GDP (Trillion US\$)")
title!(p1[1], "GDP Dynamics")

# Temperature (T) - Temperature Anomaly in °C
plot!(p1[2], years_train, T_train_original,
    seriestype=:scatter, label="Train",
    markersize=5, alpha=0.8, color=:blue)
plot!(p1[2], years_test, T_test_original,
    seriestype=:scatter, label="Test",
    markersize=5, alpha=0.8, color=:green, markershape=:diamond)
plot!(p1[2], years, T_pred_original,
    label="UDE Model (Continuous)", linewidth=2.5, color=:red)
vline!(p1[2], [train_end_year], linestyle=:dash, color=:gray, linewidth=2, label="")
xlabel!(p1[2], "Year")
ylabel!(p1[2], "Temp Anomaly (°C)")
title!(p1[2], "Temperature Dynamics")

# CO2 (C) - CO2 Concentration in ppm
plot!(p1[3], years_train, C_train_original,
    seriestype=:scatter, label="Train",
    markersize=5, alpha=0.8, color=:blue)
plot!(p1[3], years_test, C_test_original,
    seriestype=:scatter, label="Test",
    markersize=5, alpha=0.8, color=:green, markershape=:diamond)
plot!(p1[3], years, C_pred_original,
    label="UDE Model (Continuous)", linewidth=2.5, color=:red)
vline!(p1[3], [train_end_year], linestyle=:dash, color=:gray, linewidth=2, label="")
xlabel!(p1[3], "Year")
ylabel!(p1[3], "CO₂ (ppm)")
title!(p1[3], "CO₂ Dynamics")

# Capital (K) - Capital Stock in Trillion 2021US$
plot!(p1[4], years_train, K_train_original,
    seriestype=:scatter, label="Train",
    markersize=5, alpha=0.8, color=:blue)
plot!(p1[4], years_test, K_test_original,
    seriestype=:scatter, label="Test",
    markersize=5, alpha=0.8, color=:green, markershape=:diamond)
plot!(p1[4], years, K_pred_original,
    label="UDE Model (Continuous)", linewidth=2.5, color=:red)
vline!(p1[4], [train_end_year], linestyle=:dash, color=:gray, linewidth=2, label="")
xlabel!(p1[4], "Year")
ylabel!(p1[4], "Capital (Trillion US\$)")
title!(p1[4], "Capital Dynamics")

# Save with descriptive filename including CONTINUOUS approach
output_filename = "ude_continuous_adam+adamw+adam_train$(train_end_year)_test$(test_end_year)_v5.png"
savefig(p1, joinpath(@__DIR__, output_filename))
println("✓ Saved: $(output_filename)")

# --- Plot 2: Training Loss ---
println("Creating Training Loss plot...")

p2 = plot(1:length(loss_history), loss_history,
    xlabel="Iteration",
    ylabel="Continuous L2 Loss",
    title="Training Loss (CONTINUOUS) | $(opt_strategy) | $(total_iters) iters",
    linewidth=2,
    color=:green,
    legend=false,
    yscale=:log10,
    size=(1000, 500),
    left_margin=10Plots.mm)

# Mark stage transitions
vline!(p2, [adam_iters_1], linestyle=:dash, color=:orange, linewidth=1.5, label="Stage 1→2")
vline!(p2, [adam_iters_1 + adam_iters_2], linestyle=:dash, color=:red, linewidth=1.5, label="Stage 2→3")

loss_filename = "ude_continuous_adam+adamw+adam_train$(train_end_year)_loss_v5.png"
savefig(p2, joinpath(@__DIR__, loss_filename))
println("✓ Saved: $(loss_filename)")

# --- Plot 3: Learned NN Dynamics vs Theoretical log(C) ---
println("Creating Hidden Dynamics comparison plot...")

# Generate range of C values to evaluate
C_range = range(minimum(C_data), maximum(C_data), length=100)

# Compute NN output for each C value
nn_outputs = Float64[]
for c in C_range
    C_input = reshape([c], 1, 1)
    out, _ = nn_forcing(C_input, p_final.nn, st_nn)
    push!(nn_outputs, out[1])
end

# Compute theoretical log term
log_theoretical = log.(C_range ./ C0_ref)

# Fit a linear scaling to compare shapes
# Find a, b such that NN(C) ≈ a * log(C/C0) + b
# Using least squares
A = hcat(log_theoretical, ones(length(log_theoretical)))
coeffs = A \ nn_outputs
a_fit, b_fit = coeffs
log_scaled = a_fit .* log_theoretical .+ b_fit

p3 = plot(C_range, nn_outputs,
    label="Learned NN(C)",
    xlabel="C (CO₂ concentration)",
    ylabel="Forcing Term",
    title="Hidden Dynamics (CONTINUOUS): NN(C) vs Theoretical ln(C/C₀)\\n(Scaled for comparison: NN ≈ $(round(a_fit, digits=2))·ln(C/C₀) + $(round(b_fit, digits=2)))",
    linewidth=3,
    color=:red,
    size=(900, 550),
    legend=:topleft)

plot!(p3, C_range, log_scaled,
    label="Theoretical: $(round(a_fit, digits=2))·ln(C/C₀) + $(round(b_fit, digits=2))",
    linewidth=2.5,
    linestyle=:dash,
    color=:blue)

# Also plot the raw log for reference
plot!(p3, C_range, log_theoretical,
    label="Raw ln(C/C₀)",
    linewidth=1.5,
    linestyle=:dot,
    color=:gray,
    alpha=0.6)

savefig(p3, joinpath(@__DIR__, "plot_hidden_dynamics_continuous_v5.png"))
println("✓ Saved: plot_hidden_dynamics_continuous_v5.png")

# --- Plot 4: Comparison of Continuous Interpolation vs Discrete Data ---
println("Creating Interpolation Comparison plot...")

# Generate fine time grid
t_fine = range(t_train[1], t_train[end], length=500)
years_fine = years_train[1] .+ t_fine

# Evaluate interpolants at fine grid
Y_interp_fine = [Y_interp(t) for t in t_fine] .* Y_1959 ./ 1e6
T_interp_fine = [T_interp(t) for t in t_fine] .+ T_1959
C_interp_fine = [C_interp(t) for t in t_fine] .* C_1959
K_interp_fine = [K_interp(t) for t in t_fine] .* K_1959 ./ 1e6

p4 = plot(layout=(2, 2), size=(1400, 1000),
    plot_title="Data Interpolation: Continuous vs Discrete Representation")

plot!(p4[1], years_train, Y_train_original,
    seriestype=:scatter, label="Discrete Data", markersize=4, color=:blue)
plot!(p4[1], years_fine, Y_interp_fine,
    label="CubicSpline", linewidth=2, color=:red, alpha=0.8)
xlabel!(p4[1], "Year")
ylabel!(p4[1], "GDP (Trillion US\$)")
title!(p4[1], "GDP Interpolation")

plot!(p4[2], years_train, T_train_original,
    seriestype=:scatter, label="Discrete Data", markersize=4, color=:blue)
plot!(p4[2], years_fine, T_interp_fine,
    label="CubicSpline", linewidth=2, color=:red, alpha=0.8)
xlabel!(p4[2], "Year")
ylabel!(p4[2], "Temp Anomaly (°C)")
title!(p4[2], "Temperature Interpolation")

plot!(p4[3], years_train, C_train_original,
    seriestype=:scatter, label="Discrete Data", markersize=4, color=:blue)
plot!(p4[3], years_fine, C_interp_fine,
    label="CubicSpline", linewidth=2, color=:red, alpha=0.8)
xlabel!(p4[3], "Year")
ylabel!(p4[3], "CO₂ (ppm)")
title!(p4[3], "CO₂ Interpolation")

plot!(p4[4], years_train, K_train_original,
    seriestype=:scatter, label="Discrete Data", markersize=4, color=:blue)
plot!(p4[4], years_fine, K_interp_fine,
    label="CubicSpline", linewidth=2, color=:red, alpha=0.8)
xlabel!(p4[4], "Year")
ylabel!(p4[4], "Capital (Trillion US\$)")
title!(p4[4], "Capital Interpolation")

savefig(p4, joinpath(@__DIR__, "plot_interpolation_comparison_continuous_v5.png"))
println("✓ Saved: plot_interpolation_comparison_continuous_v5.png")

# ==============================================================================
# 15. SUMMARY
# ==============================================================================
println("\n" * "="^60)
println("TRAINING COMPLETE - SUMMARY (CONTINUOUS APPROACH)")
println("="^60)

println("\n📊 Train-Test Split Configuration:")
println("  Training Period: 1959-$(train_end_year) ($(n_train) discrete points)")
println("  Test Period:     $(train_end_year+1)-$(test_end_year) ($(length(test_idx)) discrete points)")
println("  Interpolation:   CubicSpline (C2 continuous)")
println("  Integration Pts: $(n_integration_points)")

println("\n📊 Performance Metrics:")
println("  Initial Loss (Continuous): $(round(initial_loss, digits=6))")
println("  Final Loss (Continuous):   $(round(loss_final, digits=6))")
println("  Improvement:  $(round((1 - loss_final/initial_loss) * 100, digits=2))%")

println("\n🧪 Optimized Physical Parameters:")
for (i, name) in enumerate(param_names)
    val = p_final.phys[i]
    println("  $name = $(round(val, digits=6))")
end

println("\n📈 Generated Plots:")
println("  1. $(output_filename) - Train vs Test predictions (CONTINUOUS)")
println("  2. $(loss_filename)                - Loss over iterations (CONTINUOUS)")
println("  3. plot_hidden_dynamics_continuous_v5.png   - NN(C) vs theoretical log(C)")
println("  4. plot_interpolation_comparison_continuous_v5.png - Interpolation visualization")

println("\n✓ All tasks completed successfully!")
