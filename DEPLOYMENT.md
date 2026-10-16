# 🚀 GUÍA PASO A PASO: DESPLIEGUE ONLINE DE LA APLICACIÓN

Esta guía te muestra exactamente cómo poner tu aplicación en una **URL pública online accesible desde cualquier celular, computadora o desde la mesa de examen**, cumpliendo con la exigencia de la rúbrica oficial:
> *"Repo y app accesibles desde la mesa: URL pública en el README, repo público. Demo en vivo obligatoria, preferentemente contra la app desplegada."*

El proyecto ya quedó configurado y optimizado con:
* [x] **`Dockerfile`**: Optimizado con PyTorch CPU (descarga rápida y bajo consumo de memoria).
* [x] **`render.yaml`**: Configuración lista para despliegue en 1 clic en Render.com.
* [x] **`.dockerignore`**: Excluye cachés y archivos innecesarios para agilizar la compilación.

---

## 🏆 OPCIÓN 1 (LA MÁS RECOMENDADA): Hugging Face Spaces

**¿Por qué es la mejor opción?**
* **Es 100% gratuita y específica para Machine Learning.**
* Te da **16 GB de memoria RAM** y **2 vCPU** de regalo (mucho más que Render, que da 512 MB). Nunca vas a tener problemas de falta de memoria al cargar el modelo.
* Te genera una URL permanente HTTPS (ej: `https://huggingface.co/spaces/TuUsuario/yugioh-identifier`).

### Pasos para desplegar en Hugging Face Spaces (5 minutos):

1. **Creá tu cuenta:** Entrá a [huggingface.co](https://huggingface.co) y registrate (es gratis).
2. **Creá un nuevo Space:**
   * Hacé clic en tu foto de perfil (arriba a la derecha) $\to$ **New Space**.
   * **Space name:** `yugioh-card-identifier` (o el nombre que prefieras).
   * **License:** `mit` o `apache-2.0`.
   * **Select the Space SDK:** Elegí **Docker** $\to$ **Blank**.
   * **Space hardware:** `CPU basic · 2 vCPU · 16 GB · Free`.
   * Hacé clic en **Create Space**.
3. **Subí el código:**
   * En la página que se abre, Hugging Face te da las instrucciones para clonar y subir por Git, o podés sincronizarlo directo con tu repositorio de GitHub:
     * Si usás Git:
       ```bash
       git remote add space https://huggingface.co/spaces/TU_USUARIO/yugioh-card-identifier
       git push space main
       ```
   * En 3 a 5 minutos compila el `Dockerfile` automáticamente y tu app queda funcionando online en la URL pública.

---

## 🌐 OPCIÓN 2: Render.com (Conexión directa a GitHub)

Render es la plataforma web más conocida para proyectos universitarios. Se conecta directo a tu cuenta de GitHub y se actualiza sola cada vez que hacés un `git push`.

### Pasos para desplegar en Render:

1. **Creá tu cuenta:** Entrá a [render.com](https://render.com) e iniciá sesión con tu cuenta de **GitHub**.
2. **Creá un Web Service:**
   * En el dashboard, hacé clic en **New +** $\to$ **Web Service**.
   * Seleccioná **Build and deploy from a Git repository**.
   * Elegí tu repositorio: `AguExposito/Yugi-RedesNeuronales` (hacé clic en **Connect**).
3. **Configuración del servicio:**
   * **Name:** `yugi-redes-neuronales`
   * **Region:** Ohio (US East) o Frankfurt.
   * **Runtime:** Seleccioná **Docker** (Render detectará automáticamente el archivo `Dockerfile` del proyecto).
   * **Instance Type:** **Free** ($0 / month).
4. **Hacé clic en "Create Web Service":**
   * Render construirá la imagen Docker e iniciará el servidor FastAPI.
   * En 3 a 4 minutos te entregará una URL pública como:  
     `https://yugi-redes-neuronales.onrender.com`

> **Nota sobre el plan gratuito de Render:** Si la app pasa 15 minutos sin recibir visitas, se "duerme". Cuando alguien entra, tarda unos 40 segundos en despertar. Por eso, el día de la presentación abrila 5 minutos antes de empezar a hablar.

---

## ⚡ OPCIÓN 3 (EL AS BAJO LA MANGA PARA LA DEMO): Cloudflare Tunnel o ngrok

Si el día de la defensa el wifi del aula tiene bloqueos de red o querés asegurarte de que la app responda con la máxima velocidad de tu propia notebook sin depender de la nube, podés usar **Cloudflare Tunnels** o **ngrok**.

Esto te da una **URL pública con HTTPS en 5 segundos** apuntando a tu servidor local:

### Con Cloudflare (Sin registro y gratis):
1. Descargás el ejecutable `cloudflared.exe` de Cloudflare.
2. En una terminal levantás tu app local:
   ```bash
   uvicorn backend.main:app --port 8000
   ```
3. En otra terminal ejecutás:
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
4. Te imprime una URL como: `https://palabras-al-azar.trycloudflare.com`.
5. Los profesores entran desde su celular a esa URL y están consumiendo tu modelo directamente con latencia de 0.14 ms.

---

## 📋 Checklist Final para el Examen
* [ ] La app responde en la URL pública.
* [ ] Pegaste la URL pública en la primera sección del `README.md`.
* [ ] Probaste subir una foto desde el celular conectado a datos móviles.
* [ ] Tenés abierta la versión local (`localhost:8000`) como respaldo por si falla la conexión a internet en la facultad.
