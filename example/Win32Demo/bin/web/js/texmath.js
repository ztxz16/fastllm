/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Stefan Goessner - 2017-22. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/
'use strict';

function escapeHTML(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function texmath(md, options) {
    const delimiters = texmath.mergeDelimiters(options && options.delimiters);
    const outerSpace = options && options.outerSpace || false;         // inline rules, effectively `dollars` require surrounding spaces, i.e ` $\psi$ `, to be accepted as inline formulas. This is primarily a guard against misinterpreting single `$`'s in normal markdown text (relevant for inline math only. Default: `false`, for backwards compatibility).
    const katexOptions = options && options.katexOptions || {};
    katexOptions.throwOnError = katexOptions.throwOnError || false; 
    katexOptions.macros = katexOptions.macros || options && options.macros;  // ensure backwards compatibility

    if (!texmath.katex) { // else ... deprecated `use` method was used ...
        if (options && typeof options.engine === 'object') {
            texmath.katex = options.engine;
        }
        else if (typeof module === "object")
            texmath.katex = require('katex');
        else  // artifical error object.
            texmath.katex = { renderToString() { return 'No math renderer found.' } };
    }

    // inject inline rules to markdown-it
    for (const rule of delimiters.inline) {
        if (!!outerSpace && 'outerSpace' in rule) rule.outerSpace = true;
        md.inline.ruler.before('escape', rule.name, texmath.inline(rule));  // ! important
        md.renderer.rules[rule.name] = (tokens, idx) => rule.tmpl.replace(/\$1/,texmath.render(tokens[idx].content,!!rule.displayMode,katexOptions));
    }
    // inject block rules to markdown-it
    for (const rule of delimiters.block) {
        md.block.ruler.before('fence', rule.name, texmath.block(rule));  // ! important for ```math delimiters
        md.renderer.rules[rule.name] = (tokens, idx) => rule.tmpl.replace(/\$2/,escapeHTML(tokens[idx].info))  // equation number .. ?
                                                                 .replace(/\$1/,texmath.render(tokens[idx].content,true,katexOptions));
    }
}

texmath.mergeDelimiters = function(delims) {
    const delimsArr = Array.isArray(delims) ? delims 
                    : typeof delims === "string" ? [delims]
                    : ['dollars'];
    const delimiters = { inline:[], block:[]};  // target of merge process ...

    for (const delim of delimsArr)  // merge them into delimiters ...
        if (delim in texmath.rules) {
            delimiters.inline.push(...texmath.rules[delim].inline);
            delimiters.block.push(...texmath.rules[delim].block);
        }

    return delimiters;
}

// texmath.inline = (rule) => dollar;  // just for debugging/testing ..

texmath.inline = (rule) => 
    function(state, silent) {
        const pos = state.pos;
        const str = state.src;
        const pre = str.startsWith(rule.tag, rule.rex.lastIndex = pos) && (!rule.pre || rule.pre(str, rule.outerSpace, pos));  // valid pre-condition ...
        const match = pre && rule.rex.exec(str);
        const res = !!match && pos < rule.rex.lastIndex && (!rule.post || rule.post(str, rule.outerSpace, rule.rex.lastIndex - 1));

        if (res) { 
            if (!silent) {
                const token = state.push(rule.name, 'math', 0);
                token.content = match[1];
                token.markup = rule.tag;
            }
            state.pos = rule.rex.lastIndex;
        }
        return res;
    }

texmath.block = (rule) => 
    function block(state, begLine, endLine, silent) {
        const pos = state.bMarks[begLine] + state.tShift[begLine];
        const str = state.src;
        const pre = str.startsWith(rule.tag, rule.rex.lastIndex = pos) && (!rule.pre || rule.pre(str, false, pos));  // valid pre-condition ....
        const match = pre && rule.rex.exec(str);
        const res = !!match
                 && pos < rule.rex.lastIndex 
                 && (!rule.post || rule.post(str, false, rule.rex.lastIndex - 1));

        if (res && !silent) {    // match and valid post-condition ...
            const endpos = rule.rex.lastIndex - 1;
            let curline;

            for (curline = begLine; curline < endLine; curline++)
                if (endpos >= state.bMarks[curline] + state.tShift[curline] && endpos <= state.eMarks[curline]) // line for end of block math found ...
                    break;

            // "this will prevent lazy continuations from ever going past our end marker"
            // s. https://github.com/markdown-it/markdown-it-container/blob/master/index.js
            const lineMax = state.lineMax;
            const parentType = state.parentType;
            state.lineMax = curline;
            state.parentType = 'math';

            if (parentType === 'blockquote') // remove all leading '>' inside multiline formula
                match[1] = match[1].replace(/(\n*?^(?:\s*>)+)/gm,'');
            // begin token
            let token = state.push(rule.name, 'math', 0);  // 'math_block'
            token.block = true;
            token.tag = rule.tag;
            token.markup = '';
            token.content = match[1];
            token.info = match[match.length-1];    // eq.no
            token.map = [ begLine, curline+1 ];
//            token.hidden = true;
            // end token ... superfluous ...

            state.parentType = parentType;
            state.lineMax = lineMax;
            state.line = curline+1;
        }
        return res;
    }

texmath.render = function(tex,displayMode,options) {
    options.displayMode = displayMode;
    let res;
    try {
        res = texmath.katex.renderToString(tex, options);
    }
    catch(err) {
        res = escapeHTML(`${tex}:${err.message}`)
    }
    return res;
}

// ! deprecated ... use options !
texmath.use = function(katex) {  // math renderer used ...
    texmath.katex = katex;       // ... katex solely at current ...
    return texmath;
}

/*
function dollar(state, silent) {
  var start, max, marker, matchStart, matchEnd, token,
      pos = state.pos,
      ch = state.src.charCodeAt(pos);

  if (ch !== 0x24) { return false; }  // $

  start = pos;
  pos++;
  max = state.posMax;

  while (pos < max && state.src.charCodeAt(pos) === 0x24) { pos++; }

  marker = state.src.slice(start, pos);

  matchStart = matchEnd = pos;

  while ((matchStart = state.src.indexOf('$', matchEnd)) !== -1) {
    matchEnd = matchStart + 1;

    while (matchEnd < max && state.src.charCodeAt(matchEnd) === 0x24) { matchEnd++; }

    if (matchEnd - matchStart === marker.length) {
      if (!silent) {
        token         = state.push('math_inline', 'math', 0);
        token.markup  = marker;
        token.content = state.src.slice(pos, matchStart)
                                 .replace(/[ \n]+/g, ' ')
                                 .trim();
      }
      state.pos = matchEnd;
      return true;
    }
  }

  if (!silent) { state.pending += marker; }
  state.pos += marker.length;
  return true;
};
*/

// used for enable/disable math rendering by `markdown-it`
texmath.inlineRuleNames = ['math_inline','math_inline_double'];
texmath.blockRuleNames  = ['math_block','math_block_eqno'];

texmath.$_pre = (str,outerSpace,beg) => {
    const prv = beg > 0 ? str[beg-1].charCodeAt(0) : false;
    return outerSpace ? !prv || prv === 0x20           // space  (avoiding regex's for performance reasons)
                      : !prv || prv !== 0x5c           // no backslash,
                             && (prv < 0x30 || prv > 0x39); // no decimal digit .. before opening '$'
}
texmath.$_post = (str,outerSpace,end) => {
    const nxt = str[end+1] && str[end+1].charCodeAt(0);
    return outerSpace ? !nxt || nxt === 0x20           // space  (avoiding regex's for performance reasons)
                             || nxt === 0x2e           // '.'
                             || nxt === 0x2c           // ','
                             || nxt === 0x3b           // ';'
                      : !nxt || nxt < 0x30 || nxt > 0x39;   // no decimal digit .. after closing '$'
}

texmath.rules = {
    brackets: {
        inline: [ 
            {   name: 'math_inline',
                rex: /\\\((.+?)\\\)/gy,
                tmpl: '<eq>$1</eq>',
                tag: '\\('
            }
        ],
        block: [
            {   name: 'math_block_eqno',
                rex: /\\\[(((?!\\\]|\\\[)[\s\S])+?)\\\]\s*?\(([^)$\r\n]+?)\)/gmy,
                tmpl: '<section class="eqno"><eqn>$1</eqn><span>($2)</span></section>',
                tag: '\\['
            },
            {   name: 'math_block',
                rex: /\\\[([\s\S]+?)\\\]/gmy,
                tmpl: '<section><eqn>$1</eqn></section>',
                tag: '\\['
            }
        ]
    },
    doxygen: {
        inline: [ 
            {   name: 'math_inline', 
                rex: /\\f\$(.+?)\\f\$/gy,
                tmpl: '<eq>$1</eq>',
                tag: '\\f$'
            }
        ],
        block: [
            {   name: 'math_block_eqno',
                rex: /\\f\[([^]+?)\\f\]\s*?\(([^)\s]+?)\)/gmy,
                tmpl: '<section class="eqno"><eqn>$1</eqn><span>($2)</span></section>',
                tag: '\\f['
            },
            {   name: 'math_block',
                rex: /\\f\[([^]+?)\\f\]/gmy,
                tmpl: '<section><eqn>$1</eqn></section>',
                tag: '\\f['
            }
        ]
    },
    gitlab: {
        inline: [ 
            {   name: 'math_inline',
                rex: /\$`(.+?)`\$/gy,
                tmpl: '<eq>$1</eq>',
                tag: '$`'
            }
        ],
        block: [
            {   name: 'math_block_eqno',
                rex: /`{3}math\s*([^`]+?)\s*?`{3}\s*\(([^)\r\n]+?)\)/gm,
                tmpl: '<section class="eqno"><eqn>$1</eqn><span>($2)</span></section>',
                tag: '```math'
            },
            {   name: 'math_block',
                rex: /`{3}math\s*([^`]*?)\s*`{3}/gm,
                tmpl: '<section><eqn>$1</eqn></section>',
                tag: '```math'
            }
        ]
    },
    julia: {
        inline: [ 
            {   name: 'math_inline', 
                rex: /`{2}([^`]+?)`{2}/gy,
                tmpl: '<eq>$1</eq>',
                tag: '``'
            },
            {   name: 'math_inline',
                rex: /\$((?:\S?)|(?:\S.*?\S))\$/gy,
                tmpl: '<eq>$1</eq>',
                tag: '$',
                spaceEnclosed: false,
                pre: texmath.$_pre,
                post: texmath.$_post,

            }
        ],
        block: [
            {   name: 'math_block_eqno',
                rex: /`{3}math\s+?([^`]+?)\s+?`{3}\s*?\(([^)$\r\n]+?)\)/gmy,
                tmpl: '<section class="eqno"><eqn>$1</eqn><span>($2)</span></section>',
                tag: '```math'
            },
            {   name: 'math_block',
                rex: /`{3}math\s+?([^`]+?)\s+?`{3}/gmy,
                tmpl: '<section><eqn>$1</eqn></section>',
                tag: '```math'
            }
        ]
    },
    kramdown: {
        inline: [ 
            {   name: 'math_inline', 
                rex: /\${2}(.+?)\${2}/gy,
                tmpl: '<eq>$1</eq>',
                tag: '$$'
            }
        ],
        block: [
            {   name: 'math_block_eqno',
                rex: /\${2}([^$]+?)\${2}\s*?\(([^)\s]+?)\)/gmy,
                tmpl: '<section class="eqno"><eqn>$1</eqn><span>($2)</span></section>',
                tag: '$$'
            },
            {   name: 'math_block',
                rex: /\${2}([^$]+?)\${2}/gmy,
                tmpl: '<section><eqn>$1</eqn></section>',
                tag: '$$'
            }
        ]
    },
    beg_end: {
        inline: [],
        block: [
            {
                name: "math_block",
                rex: /(\\(?:begin)\{([a-z]+)\}[\s\S]+?\\(?:end)\{\2\})/gmy, // regexp to match \begin{...}...\end{...} environment.
                tmpl: "<section><eqn>$1</eqn></section>",
                tag: "\\"
            }
        ]
    },
    dollars: {
        inline: [
            {   name: 'math_inline_double',
                rex: /\${2}([^$]*?[^\\])\${2}/gy,
                tmpl: '<section><eqn>$1</eqn></section>',
                tag: '$$',
                displayMode: true,
                pre: texmath.$_pre,
                post: texmath.$_post
            },
            {   name: 'math_inline',
                rex: /\$((?:[^\s\\])|(?:\S.*?[^\s\\]))\$/gy,
                tmpl: '<eq>$1</eq>',
                tag: '$',
                outerSpace: false,
                pre: texmath.$_pre,
                post: texmath.$_post
            }
        ],
        block: [
            {   name: 'math_block_eqno',
                rex: /\${2}([^$]*?[^\\])\${2}\s*?\(([^)\s]+?)\)/gmy,
                tmpl: '<section class="eqno"><eqn>$1</eqn><span>($2)</span></section>',
                tag: '$$'
            },
            {   name: 'math_block',
                rex: /\${2}([^$]*?[^\\])\${2}/gmy,
                tmpl: '<section><eqn>$1</eqn></section>',
                tag: '$$'
            }
        ]
    }
};

if (typeof module === "object" && module.exports)
   module.exports = texmath;
