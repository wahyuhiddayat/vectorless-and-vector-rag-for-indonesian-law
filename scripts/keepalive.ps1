# Keep Windows from sleeping during long-running eval.
#
# Usage:
#   In a SEPARATE PowerShell window (not the eval terminal), run:
#       .\scripts\keepalive.ps1
#   Leave the window open. Press Ctrl+C to release the keepalive.
#
# This calls SetThreadExecutionState with ES_CONTINUOUS | ES_SYSTEM_REQUIRED
# | ES_DISPLAY_REQUIRED, refreshed every 30 seconds. The flag persists only
# while this PowerShell process is alive, so closing the window or Ctrl+C
# releases the system back to its normal power policy.
#
# Does NOT require admin. Does NOT modify the global power plan. Safer than
# `powercfg /change` because it auto-resets on process exit.

$signature = @"
[DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
"@
$power = Add-Type -MemberDefinition $signature -Name SystemPower -Namespace Win32 -PassThru

# ES_CONTINUOUS         = 0x80000000  -- persistent until reset
# ES_SYSTEM_REQUIRED    = 0x00000001  -- prevent system sleep
# ES_DISPLAY_REQUIRED   = 0x00000002  -- prevent display sleep (also implies system)
$flags = [uint32]"0x80000003"

Write-Host "[keepalive] Anti-sleep flag set. Laptop akan stay awake."
Write-Host "[keepalive] Press Ctrl+C to release and let normal power policy resume."
Write-Host ""

try {
    while ($true) {
        $null = $power::SetThreadExecutionState($flags)
        Start-Sleep -Seconds 30
    }
}
finally {
    # ES_CONTINUOUS alone (0x80000000) releases the system-required + display-required
    # flags back to default behavior.
    $null = $power::SetThreadExecutionState([uint32]"0x80000000")
    Write-Host ""
    Write-Host "[keepalive] Released. Normal power policy resumed."
}
