@{
    Severity     = @('Error', 'Warning')
    ExcludeRules = @(
        # Los scripts son CLIs interactivos; Write-Host comunica estado al usuario.
        'PSAvoidUsingWriteHost'
    )
}

