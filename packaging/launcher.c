/*
 * Utter.app's main executable.
 *
 * Why a compiled stub rather than a shell script, and why exec rather than
 * spawn: macOS attributes Accessibility and Microphone permission to the
 * *bundle* whose main executable was launched. A shell script in
 * Contents/MacOS works for launching but leaves the running process looking
 * like /bin/sh to some of the machinery, and the author has already lost an
 * evening to a permission that was granted to the terminal instead of the
 * app.
 *
 * execv keeps the process id and the launch context, so the Python that ends
 * up running is still the process macOS started from the bundle. Verified by
 * reading NSBundle.mainBundle().bundleIdentifier() from inside it — see
 * `utter bundle --check`.
 *
 * The interpreter path is compiled in rather than searched for. A tool that
 * silently picks up a different Python than the one its dependencies are
 * installed into fails in a way that takes an hour to understand.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <libgen.h>
#include <mach-o/dyld.h>
#include <spawn.h>
#include <signal.h>
#include <sys/wait.h>

extern char **environ;

static pid_t child_pid = 0;

static void forward_signal(int signum) {
    if (child_pid > 0) {
        kill(child_pid, signum);
    }
}

#ifndef UTTER_PYTHONHOME
#error "compile with -DUTTER_PYTHONHOME and -DUTTER_PYTHONPATH"
#endif

int main(int argc, char *argv[]) {
    /* Contents/MacOS/Utter -> Contents/Resources */
    char self[4096];
    uint32_t size = sizeof(self);
    if (_NSGetExecutablePath(self, &size) != 0) {
        fprintf(stderr, "Utter: cannot locate my own bundle\n");
        return 1;
    }
    char *macos_dir = dirname(self);            /* .../Contents/MacOS   */
    char contents[4096];
    snprintf(contents, sizeof(contents), "%s", dirname(macos_dir));

    char resources[4096];
    snprintf(resources, sizeof(resources), "%s/Resources", contents);
    setenv("UTTER_APP_RESOURCES", resources, 1);

    /* The load-bearing line.
     *
     * execv replaces this image with Python's, and CoreFoundation then locates
     * "the main bundle" by walking up from the *running executable* — which is
     * now ~/.venvs/utter/bin/python, outside the bundle entirely. NSBundle
     * came back nil, and a nil bundle means macOS files the Accessibility and
     * Microphone grants against Python rather than against Utter: exactly the
     * mistake that put this project's first grant on Terminal.
     *
     * CFProcessPath tells CoreFoundation where the process "is" regardless of
     * what it is running. Undocumented but long-standing, and it is how every
     * wrapper-style .app solves this. Verified with `Utter.app/.../Utter
     * --check`, which prints the bundle identifier it ends up with. */
    setenv("CFProcessPath", self, 1);

    /* The interpreter lives in the bundle, and that is the whole point.
     *
     * Pointing at ~/.venvs/utter/bin/python worked for everything except the
     * microphone: TCC judges by code signature, not by CFProcessPath, and an
     * anaconda binary outside the bundle is not this app. The permission
     * request came back refused in one millisecond with no prompt, and macOS
     * then handed the recording exact zeros rather than an error.
     *
     * A copy of the interpreter inside Contents/MacOS is signed along with the
     * bundle, so the process asking for the microphone is Utter. It needs
     * PYTHONHOME to find its standard library, since it is no longer beside
     * it, and PYTHONPATH for the venv's packages. */
    char python[4096];
    snprintf(python, sizeof(python), "%s/MacOS/python", contents);
    setenv("PYTHONHOME", UTTER_PYTHONHOME, 1);
    setenv("PYTHONPATH", UTTER_PYTHONPATH, 1);

    /* Unbuffered, so the log file is useful while the app is still running
     * rather than only after it exits. */
    setenv("PYTHONUNBUFFERED", "1", 1);

    char *args[8];
    int n = 0;
    args[n++] = python;
    args[n++] = "-m";
    args[n++] = "backend.app";
    for (int i = 1; i < argc && n < 7; i++) {
        args[n++] = argv[i];
    }
    args[n] = NULL;

    /* Spawn a child rather than exec, and wait for it.
     *
     * execv replaced this process image with Python's, and LaunchServices
     * never saw the application it started finish launching. Everything
     * worked except the one thing that needs the app to be a full citizen:
     * the status item was created, reported isVisible() == YES, and was never
     * laid out — frame stayed 32x0 at the origin, so no icon appeared. The
     * same binary run straight from a terminal was fine, which is what made
     * this take an hour to see.
     *
     * Keeping this process alive as the thing LaunchServices launched, with
     * Python as its child, gives both halves what they need. The child sets
     * CFProcessPath so it still belongs to the bundle for permissions. */
    if (getenv("UTTER_EXEC") != NULL) {
        /* exec: the process BECOMES Python, keeping this bundle's identity, so
         * macOS attributes the microphone and Accessibility grants to Utter.
         * A spawned child is a foreign binary and the microphone request is
         * refused outright — measured, 1ms, no prompt. */
        execv(python, args);
        fprintf(stderr, "Utter: could not exec %s: %s\n", python, strerror(errno));
        return 1;
    }

    pid_t child;
    if (posix_spawn(&child, python, NULL, NULL, args, environ) != 0) {
        fprintf(stderr, "Utter: could not start %s: %s\n", python, strerror(errno));
        return 1;
    }

    /* Pass on the signals launchd and the Dock use to stop an app, so quitting
     * Utter actually quits Python rather than orphaning it. */
    signal(SIGTERM, forward_signal);
    signal(SIGINT, forward_signal);
    child_pid = child;

    int status = 0;
    while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
        /* a forwarded signal interrupted the wait; keep waiting */
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
