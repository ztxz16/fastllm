/* Native file-manager entrypoint; the payload stays in the relocatable bundle. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
    size_t capacity = 256;
    char *executable = NULL;
    for (;;) {
        char *buffer = realloc(executable, capacity);
        if (!buffer) {
            free(executable);
            perror("FastLLM launcher: allocate path");
            return 1;
        }
        executable = buffer;
        ssize_t length = readlink("/proc/self/exe", executable, capacity - 1);
        if (length < 0) {
            perror("FastLLM launcher: locate executable");
            free(executable);
            return 1;
        }
        if ((size_t)length < capacity - 1) {
            executable[length] = '\0';
            break;
        }
        capacity *= 2;
    }

    char *separator = strrchr(executable, '/');
    if (!separator) {
        fprintf(stderr, "FastLLM launcher: invalid executable path\n");
        free(executable);
        return 1;
    }
    size_t root_length = (size_t)(separator - executable);
    const char suffix[] = "/support/entrypoint.sh";
    char *script = malloc(root_length + sizeof(suffix));
    char **arguments = calloc((size_t)argc + 3, sizeof(*arguments));
    if (!script || !arguments) {
        perror("FastLLM launcher: allocate arguments");
        free(arguments);
        free(script);
        free(executable);
        return 1;
    }
    memcpy(script, executable, root_length);
    memcpy(script + root_length, suffix, sizeof(suffix));
    arguments[0] = "bash";
    arguments[1] = script;
    arguments[2] = executable;
    for (int index = 1; index < argc; ++index) {
        arguments[index + 2] = argv[index];
    }
    execv("/bin/bash", arguments);
    perror("FastLLM launcher: start bundled entrypoint");
    free(arguments);
    free(script);
    free(executable);
    return 1;
}
