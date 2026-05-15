"""
measure_wigner.jl — hardware-evaluation callback for Intonato closed-loop QOC.

Calls the lab-box experiment service over the SSH tunnel:
  http://127.0.0.1:18765/run_wigner

Assumes the tunnel is already open on this machine:
  ssh -L 18765:127.0.0.1:18765 <fridge-pc>

Contract (matches job_server/closed_loop/service.py):
- IQ envelopes are drive amplitudes in GHz (same unit as Piccolo's native
  output and the npz files in device.optimal_control). Typical peaks are
  0.001-0.01 GHz (1-10 MHz Rabi). Practical upper bound is whatever produces
  a QICK gain register ≤ 32767 (roughly 0.04 GHz with current π-pulse cal).
- Times are in microseconds. Only duration matters; the DAC sample grid is
  hardware-fixed and the envelope is linearly interpolated onto it.
- Service computes per-channel QICK gains from the GHz peaks via the
  calibrated π-pulse (MM_base.get_gain_optimal_pulse math). Orchestrator
  doesn't need to know about DAC counts.

Returns (parity, alphas, iter_id, shots_path, mode, meta) as a NamedTuple.
"""
module MeasureWigner

using HTTP
using JSON3

const DEFAULT_SERVICE_URL = "http://127.0.0.1:18765"

"""
    measure_wigner(IQ_table_GHz, alphas; kwargs...) -> NamedTuple

Args
----
IQ_table_GHz :: NamedTuple/Dict with keys (times, I_c, Q_c, I_q, Q_q).
                Values in GHz Rabi; times in μs. Typical peaks 0.001-0.01 GHz.
alphas       :: Vector{ComplexF64}.

Keyword args
------------
reps         = 1000
mode         = "hw"   # "sim" for plumbing tests with no hardware
pulse_ref    = [["optimal_control", "fock", "2", [0, 0]]]
qubits       = [0]
man_mode_no  = 1
service_url  = DEFAULT_SERVICE_URL
knobs        = NamedTuple()  # merged into request defaults
gain_override = nothing       # pass (qb=N, cav=M) to pin gains
timeout      = 600.0          # seconds; covers slow acquires
"""
function measure_wigner(
    IQ_table_GHz,
    alphas::AbstractVector{<:Complex};
    reps::Int             = 1000,
    mode::String          = "hw",
    pulse_ref             = [["optimal_control", "fock", "2", [0, 0]]],
    qubits                = [0],
    man_mode_no::Int      = 1,
    service_url::String   = DEFAULT_SERVICE_URL,
    knobs                 = (;),
    gain_override         = nothing,
    timeout::Real         = 600.0,
)
    # IQ_table_GHz can be a NamedTuple or a Dict — normalize.
    iq = (
        times = collect(Float64, _field(IQ_table_GHz, :times, "times")),
        I_c   = collect(Float64, _field(IQ_table_GHz, :I_c,   "I_c")),
        Q_c   = collect(Float64, _field(IQ_table_GHz, :Q_c,   "Q_c")),
        I_q   = collect(Float64, _field(IQ_table_GHz, :I_q,   "I_q")),
        Q_q   = collect(Float64, _field(IQ_table_GHz, :Q_q,   "Q_q")),
    )

    body = Dict(
        "mode"        => mode,
        "IQ_table"    => iq,
        "alphas"      => [[real(a), imag(a)] for a in alphas],
        "reps"        => reps,
        "pulse_ref"   => pulse_ref,
        "qubits"      => qubits,
        "man_mode_no" => man_mode_no,
        "knobs"       => Dict(pairs(knobs)),
    )
    if gain_override !== nothing
        body["gain_override"] = Dict("qb" => gain_override.qb, "cav" => gain_override.cav)
    end

    r = HTTP.post(
        "$service_url/run_wigner";
        headers      = ["Content-Type" => "application/json"],
        body         = JSON3.write(body),
        readtimeout  = Int(round(timeout)),
        retry        = false,
    )
    if r.status != 200
        error("service returned status $(r.status): $(String(r.body))")
    end

    j = JSON3.read(r.body)
    return (
        parity     = Float64.(j.parity),
        alphas     = ComplexF64[a[1] + im * a[2] for a in j.alphas],
        iter_id    = String(j.iter_id),
        shots_path = j.shots_path === nothing ? nothing : String(j.shots_path),
        mode       = String(j.mode),
        meta       = j.meta,
    )
end

# tolerate either NamedTuple or Dict access
_field(nt::NamedTuple, sym::Symbol, _key::String) = getfield(nt, sym)
_field(d::AbstractDict, _sym::Symbol, key::String) = haskey(d, key) ? d[key] : d[Symbol(key)]


"""
    health(service_url=DEFAULT_SERVICE_URL) -> JSON3.Object

GET / to confirm the service is reachable.
"""
function health(service_url::String = DEFAULT_SERVICE_URL)
    r = HTTP.get("$service_url/")
    return JSON3.read(r.body)
end


"""
    echo(payload; service_url=DEFAULT_SERVICE_URL) -> JSON3.Object

POST /echo — round-trip JSON smoke test, no hardware involvement.
"""
function echo(payload; service_url::String = DEFAULT_SERVICE_URL)
    r = HTTP.post(
        "$service_url/echo";
        headers = ["Content-Type" => "application/json"],
        body    = JSON3.write(payload),
    )
    return JSON3.read(r.body)
end

end # module
