# Solución de problemas

## `ENOSPC: no space left on device` o aviso de poco espacio en `C:`

**Causa raíz**: el disco `C:` tiene poco espacio libre. Los IDEs y gestores de paquetes guardan extensiones y cachés en `C:\Users\<usuario>\AppData`.

**Solución**:

1. Libera espacio en `C:` (se recomiendan mínimo 5 GB libres).
2. Redirige las cachés de gestores de paquetes a otro disco o carpeta fuera de OneDrive:
   ```powershell
   setx NPM_CONFIG_CACHE "D:\caches\npm"
   setx PIP_CACHE_DIR "D:\caches\pip"
   setx UV_CACHE_DIR "D:\caches\uv"
   ```
3. Si el repositorio vive dentro de **OneDrive**, asegúrate de no versionar ni indexar grabaciones de audio (`.mp3`, `.wav`), modelos (`.onnx`, `.pt`, `.gguf`) ni entornos virtuales (`.venv/`, `node_modules/`). El archivo `.gitignore` y `.graphifyignore` del perfil `base`/`graphify` ya los excluyen por defecto.

## Fallo en `scripts/lint` por "Secreto detectado"

**Causa**: el escáner determinístico de secretos detectó un patrón de token o clave privada (`ghp_`, `github_pat_`, `sk-`, `AKIA`, `BEGIN PRIVATE KEY`) en algún archivo del proyecto.

**Solución**:

1. Mueve cualquier credencial a variables de entorno del sistema o a un archivo `.env` local (que está ignorado por `.gitignore`).
2. Nunca incluyas tokens reales en scripts, fixtures de prueba ni archivos de configuración versionados.
3. Si el secreto ya fue commiteado previamente, revócalo inmediatamente en el proveedor correspondiente.

## `openspec` o `graphify` no están en el PATH

**Causa**: el script `bootstrap` no se corrió con el perfil correspondiente o la sesión de terminal no se reinició después de instalar `pipx`/`npm`.

**Solución**:

- Corre `./scripts/bootstrap.ps1 -Profiles openspec,graphify` (Windows) o `./scripts/bootstrap.sh --profiles openspec,graphify` (macOS/Linux).
- Cierra y vuelve a abrir el terminal.
- Verifica con `./scripts/verify-setup.ps1`.

## Cómo instalar la extensión Codex y el plugin de GitHub

El harness nunca instala ni autentica silenciosamente extensiones vinculadas a tu cuenta personal:

1. Abre tu editor compatible e instala **Codex – OpenAI’s coding agent** (`openai.chatgpt`).
2. Inicia sesión con tu cuenta y **recarga la ventana del editor**.
3. Si elegiste `-GitHubMode codex-plugin`, ejecuta `/plugins` dentro del agente para conectar el plugin **GitHub** y autorizar tu cuenta.

## `openspec validate --changes` falla

- Asegúrate de que `openspec` está instalado (`openspec --version`).
- Revisa `openspec/changes/<cambio>/` para verificar que `proposal.md`, `design.md`, `tasks.md` y `specs/` cumplen el formato de escenarios `#### Scenario:`.
