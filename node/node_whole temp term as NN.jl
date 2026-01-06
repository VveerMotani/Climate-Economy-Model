
using DifferentialEquations, Plots
using ModelingToolkit, DiffEqFlux, Flux, Lux, Random, ComponentArrays
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using CSV, DataFrames, Statistics, SciMLSensitivity


println("Step 1: Loading Real Data...")
df = CSV.read("DATA/final_training_data.csv", DataFrame)

# We extract the 4 key variables into a Matrix
# Order: Y (GDP), T (Temp), C (CO2), K (Capital)
# We transpose (') it so rows are variables, columns are time steps
real_data = Matrix(df[:, [:Y_norm, :T_adjusted, :C_norm, :K_norm]])' 
years = df.Year
tspan = (Float64(years[1]), Float64(years[end])) 
tsteps = Float64.(years)
u0 = real_data[:, 1] 

# --- 📝 LOGGING SETUP ---
const LOG_FILE = "training_log.txt"

# Create/Overwrite the file and write the header
open(LOG_FILE, "w") do io
    println(io, "=========================================================")
    println(io, "              CLIMATE UDE TRAINING LOG                   ")
    println(io, "=========================================================")
    println(io, "Iter  | Loss       | β (Growth)| θ (Emit)  | ψ (Dam)   ")
    println(io, "------|------------|-----------|-----------|-----------")
end
println("Logging started in: $LOG_FILE")

# ------------------------------------------------------------------------------
# 2. DEFINE MODEL
# ------------------------------------------------------------------------------
rng = Random.default_rng()
Random.seed!(rng, 123) 

# Neural Network
NN = Lux.Chain(
    Lux.Dense(2, 10, tanh),
    Lux.Dense(10, 1)
)
p_nn, st_nn = Lux.setup(rng, NN)

# Physics Parameters (Initial Guesses)
p_phys_init = ComponentArray(
    β = 0.03,  # Initial guess: 3.0%
    ψ = 0.005, 
    θ = 0.5, 
    ξ = 0.02, 
    s = 0.25, 
    δ = 0.05
)

p_all = ComponentArray(phys=p_phys_init, nn=p_nn)

# Print Initial State
println("\n--- 🟢 INITIAL GUESSES (BEFORE TRAINING) ---")
println("Growth Rate (β): ", p_phys_init.β)
println("Damage (ψ):      ", p_phys_init.ψ)
println("Emissions (θ):   ", p_phys_init.θ)
println("Sink Rate (ξ):   ", p_phys_init.ξ)
println("Savings (s):     ", p_phys_init.s)
println("Depreciation (δ):", p_phys_init.δ)
println("--------------------------------------------")

# ------------------------------------------------------------------------------
# 3. ROBUST UDE DEFINITION
# ------------------------------------------------------------------------------
function climate_ude(du, u, p, t)
    Y, T, C, K = u
    β, ψ, θ, ξ, s, δ = p.phys
    
    # Safety Clamps
    Y_safe = clamp(Y, 0.1, 20.0) 
    
    du[1] = (abs(β) * Y) - (abs(ψ) * T^2 * Y)
    
    nn_input = [C, T]
    nn_output, _ = Lux.apply(NN, nn_input, p.nn, st_nn)
    du[2] = nn_output[1] 

    du[3] = (abs(θ) * Y) - (abs(ξ) * (C - 1.0))
    du[4] = (abs(s) * Y) - (abs(δ) * K)
end

prob_ude = ODEProblem(climate_ude, u0, tspan, p_all)

# ------------------------------------------------------------------------------
# 4. ROBUST TRAINING
# ------------------------------------------------------------------------------
println("\nStep 3: Training with Stiff Solver...")

function predict(theta)
    Array(solve(prob_ude, Rodas5(), p=theta, saveat=tsteps, 
                abstol=1e-5, reltol=1e-5,
                sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end

function loss_function(theta)
    pred = predict(theta)
    if size(pred) != size(real_data)
        return 1e9 
    end

    l_Y = sum(abs2, pred[1,:] .- real_data[1,:])
    l_T = sum(abs2, pred[2,:] .- real_data[2,:]) * 10.0
    l_C = sum(abs2, pred[3,:] .- real_data[3,:])
    l_K = sum(abs2, pred[4,:] .- real_data[4,:])
    
    return l_Y + l_T + l_C + l_K
end

losses = Float64[]
# --- REPLACE YOUR OLD CALLBACK WITH THIS ---

using Printf # Ensure this is at the top of your file

# --- 📝 FILE LOGGING CALLBACK ---
callback = function (p, l)
    push!(losses, l)
    
    if length(losses) % 50 == 0
        
        println("Iter: $(length(losses)) | Loss: $l")
        
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_function(x), adtype)
optprob = Optimization.OptimizationProblem(optf, p_all)

println("Phase 1: ADAM (1000 iters)...")
res1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.01), maxiters=1000, callback=callback)

println("Phase 2: BFGS (500 iters)...")
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, OptimizationOptimJL.BFGS(), maxiters=500, callback=callback)

println("Final Loss: ", res2.minimum)

# ------------------------------------------------------------------------------
# 5. FINAL REPORTING
# ------------------------------------------------------------------------------
final_phys = res2.u.phys

println("\n============================================")
println("       🧪 SCIENTIFIC DISCOVERY REPORT       ")
println("============================================")
println("Param   | Initial Guess | Learned Value | Change")
println("--------|---------------|---------------|-------")

function print_row(name, init, final)
    change = round((final - init)/init * 100, digits=1)
    # Visual indicator if it changed a lot
    marker = abs(change) > 20.0 ? "❗" : " "
    Printf = @sprintf "%-7s | %-13.4f | %-13.4f | %+.1f%% %s" name init final change marker
    println(Printf)
end
 # Needed for nice formatting

print_row("β (Gro)", p_phys_init.β, final_phys.β)
print_row("ψ (Dam)", p_phys_init.ψ, final_phys.ψ)
print_row("θ (Emi)", p_phys_init.θ, final_phys.θ)
print_row("ξ (Snk)", p_phys_init.ξ, final_phys.ξ)
print_row("s (Sav)", p_phys_init.s, final_phys.s)
print_row("δ (Dep)", p_phys_init.δ, final_phys.δ)
println("============================================")



# Generate Plots
final_solution = predict(res2.u)
p1 = plot(years, real_data[1,:], seriestype=:scatter, label="Data", title="GDP", color=:blue)
plot!(p1, years, final_solution[1,:], label="Fit", linewidth=3, color=:blue)
p2 = plot(years, real_data[2,:], seriestype=:scatter, label="Data", title="Temp", color=:red)
plot!(p2, years, final_solution[2,:], label="Fit", linewidth=3, color=:red)
p3 = plot(years, real_data[3,:], seriestype=:scatter, label="Data", title="CO2", color=:green)
plot!(p3, years, final_solution[3,:], label="Fit", linewidth=3, color=:green)
p4 = plot(years, real_data[4,:], seriestype=:scatter, label="Data", title="Capital", color=:purple)
plot!(p4, years, final_solution[4,:], label="Fit", linewidth=3, color=:purple)
p5 = plot(losses, yscale=:log10, title="Loss", label="Error", color=:black)

final_plot = plot(p1, p2, p3, p4, p5, layout=(3,2), size=(1000, 1000))
savefig(final_plot, "Final_Report.png")
display(final_plot)

# --- 📝 SAVE FINAL REPORT TO LOG ---
open(LOG_FILE, "a") do io
    println(io, "\n\n============================================")
    println(io, "       🏆 FINAL LEARNED PHYSICS       ")
    println(io, "============================================")
    println(io, "Parameter | Initial Guess | Learned Value")
    println(io, "----------|---------------|---------------")
    
    # We use 'println(io, ...)' to write to the file instead of the screen
    @printf(io, "β (Growth)| %-13.4f | %.4f\n", p_phys_init.β, final_phys.β)
    @printf(io, "θ (Emit)  | %-13.4f | %.4f\n", p_phys_init.θ, final_phys.θ)
    @printf(io, "ψ (Damage)| %-13.4f | %.4f\n", p_phys_init.ψ, final_phys.ψ)
    @printf(io, "ξ (Sink)  | %-13.4f | %.4f\n", p_phys_init.ξ, final_phys.ξ)
    @printf(io, "s (Save)  | %-13.4f | %.4f\n", p_phys_init.s, final_phys.s)
    @printf(io, "δ (Depr)  | %-13.4f | %.4f\n", p_phys_init.δ, final_phys.δ)
    println(io, "============================================")
end

println("\nTraining Complete! Check $LOG_FILE for the full report.")