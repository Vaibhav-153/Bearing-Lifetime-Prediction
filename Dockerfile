# ==============================================================================
# PHASE 3: DOCKERFILE FOR CONTAINERIZATION (CORRECTED VERSION)
# ==============================================================================

# --- Stage 1: Set up the Python Environment ---
FROM python:3.11-slim

# --- Stage 2: Set the Working Directory ---
WORKDIR /app

# --- Stage 3: Install Dependencies ---
# Copy ONLY the requirements file first for build-caching
COPY ./bearing_app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 4: Copy the Application Code ---
#
# THIS IS THE CRITICAL FIX:
# The trailing slash on './bearing_app/' copies the CONTENTS of the folder,
# not the folder itself. This will place main.py, prognosticator.py, etc.
# directly into /app.
#
COPY ./bearing_app/ /app/

# --- Stage 5: Expose the Port ---
EXPOSE 8000

# --- Stage 6: Define the Startup Command ---
# Now that main.py is in /app, this command will work correctly.
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]