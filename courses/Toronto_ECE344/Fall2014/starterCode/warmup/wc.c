#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "wc.h"

struct wc {
	/* you can define this struct to have whatever fields you want. */
};

struct wc *
wc_init()
{
	struct wc *wc;

	wc = (struct wc *)malloc(sizeof(struct wc));
	assert(wc);

	TBD();

	return wc;
}

int
wc_insert_word(struct wc *wc, char *word)
{
	TBD();
	return 0;
}

void
wc_output(struct wc *wc)
{
	TBD();
}
