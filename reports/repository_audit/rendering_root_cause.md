# Rendering root cause

The failed attempt forced `--no-rendering`, `--minimize`, Qt scale `0.01`, software OpenGL/GLES, and then hit framebuffer allocation errors. The repaired launcher uses normal Windows Qt scaling, hardware rendering, no rendering-disable flags, and client-area pixel validation.
