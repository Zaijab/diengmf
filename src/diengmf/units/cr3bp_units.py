# Real time to CR3BP time conversion
TU_seconds = 382981  # Time unit in seconds
LU_kilometeers = 389703
TU_days = TU_seconds / (24 * 3600)  # ≈ 4.35 days

def real_time_to_cr3bp(real_seconds):
    return real_seconds / TU_seconds

def cr3bp_time_to_real(cr3bp_time):
    return cr3bp_time * TU_seconds
