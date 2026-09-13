<div align="center">

# Agent Configuration

</div>

- Keep an eye on [Qwen 3.8 Flash Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- Monitor Codex usage [here](https://chatgpt.com/codex/cloud/settings/analytics) because OpenAI makes it impossible to find this page.
- [This person](https://mia-ai.carrd.co/) seems vaguely interesting.

## Overview

## Pi Extensions

[This video](https://www.youtube.com/watch?v=iKwPaB5TUdI) has some cool ideas ([config](https://github.com/amosblomqvist/pi-config).

- [`npm:pi-subagents`](https://pi.dev/packages/pi-subagents)
- [`npm:pi-exa`](https://pi.dev/packages/pi-exa) maybe swap for YALMI Search when it's done ;)

### TODO

I like [`pi-interactive-subagents`](https://github.com/HazAT/pi-interactive-subagents)'s integration with terminal multiplexers.

[`pi-observational-memory`](https://github.com/HazAT/pi-interactive-subagents) seems like a good idea, but I feel like bad compaction is kind of a skill issue tbh.

## Agent Skills

- Source of skills might be [skills.sh](https://skills.sh)

### `code-review` Skill

The `code-review` skill is Based on `[code-review](https://github.com/mattpocock/skills/blob/main/skills/engineering/code-review/SKILL.md)` from [Matt Pocock's skills](https://github.com/mattpocock) as well as `[receiving-code-review](https://github.com/obra/superpowers/blob/main/skills/receiving-code-review/SKILL.md)` and `[requesting-code-review](https://github.com/obra/superpowers/blob/main/skills/requesting-code-review/SKILL.md)` from [obra/superpowers](https://github.com/obra/superpowers). Superpowers' `[code-reviewer template](https://github.com/obra/superpowers/blob/main/skills/requesting-code-review/code-reviewer.md)` is also an important reference.

### `using-workflows` Skill (Currently provided by `council-mode` and `pi-subagents`

I want to understand and potentially simplify/cut `council-mode` and `pi-subagents` provided by the `[pi-subagents](https://github.com/nicobailon/pi-subagents/)` plugin that I'm using. I don't love the way that the plugin doesn't really provide a good way to write my own skills for use with it.

I generally have liked the results from having used Superpowers' `brainstorming`/`writing-plans` followed by `[subagent-driven-development](https://github.com/obra/superpowers/tree/main/skills/subagent-driven-development)`. My main gripe was that agents would do things like say "Oh, this is simple" or with the effective meaning of "How could I possibly mess this up?" and skip reviews, combine steps, etc. `subagent-driven-development` basically demands that the LLM have the discipline to do everything right all the time, whereas a workflow-based approach would instead require the agent write a concrete, "deterministic" (insofar as agentic coding can be) workflow that loops implementation -> review -> fix -> review *once*.

It might also be prudent to include some aspects of Superpowers' `[dispatching-parallel-subagents](https://github.com/obra/superpowers/blob/main/skills/dispatching-parallel-agents/SKILL.md)`, too. Draft for new version [on rentry](https://rentry.co/using-workflows)

### Wizard Skill

Taken straight from [Matt Pocock](https://github.com/mattpocock/skills/tree/main/skills/engineering/wizard). Might move it over to Python/uv/uvx or TypeScript/npx/pnpm/bunx because Bash is a major pain to work with and DX for agents feels slightly less unimportant than I think people give it credit for.

### `writing-tests` Skill

The GPT-5.6 series really *loves* writing tests, and tbh it's annoying because most of them suck. I asked GPT-5.6 Luna what makes good tests, and ofc it wrote [way too fucking much](https://rentry.co/5_6-luna-writing-tests), but there's probably something useful there. I also asked [Kimi K2.5](https://rentry.co/kimi-k2_5-writing-tests) because I thought K2's fantastic prose might translate to it being less dogshit than other models at writing skills. Turns out K2.5/2.6/2.7 basically lost that ability and I'm probably wrong anyway.

### TODO

#### Add `brainstorming`/`grill-me`/`to-spec` Workflow/Skill

I'm looking at adding `[grilling](https://github.com/mattpocock/skills/blob/main/skills/productivity/grilling/SKILL.md)`, `[to-spec](https://github.com/mattpocock/skills/tree/main/skills/engineering/to-spec)`, and `[wayfinder](https://github.com/mattpocock/skills/blob/main/skills/engineering/wayfinder/SKILL.md)` from Matt Pocock, but I'm unsure of exactly what I want from it right now. The goal would be to get a workflow that is comparable to what you'd get from `[brainstorming](https://github.com/obra/superpowers/blob/main/skills/brainstorming/SKILL.md)` and `[writing-plans](https://github.com/obra/superpowers/blob/main/skills/writing-plans/SKILL.md)` from obra/superpowers but less rigid. I generally dislike Superpowers' "strong-arming" the agent into doing things in a very specific way. I do really like the minimalism and practicality of using Pocock's [original `grill-me` skill](https://youtu.be/EJyuu6zlQCg) (non-agent-incocable) to get a solid design down, then using `to-spec` to write it to a markdown file for later. `wayfinder` might come in handy for very large scoped work. [Had a draft](https://rentry.co/idea-distillation-draft), but it probably won't actually see use.

#### Adding a "Ponytail" Skill

Sergei Suvorin on Linkedin: [Ponytail Skill](https://lnkd.in/p/gn5YPTW9)

There's a [pi extension](https://pi.dev/packages/@dietrichgebert/ponytail) for it??

#### Unslop

Instruct agent to speak in ASD-STE100?

Maybe poteto/pstack's [unslop skill](https://github.com/cursor/plugins/blob/main/pstack/skills/unslop/SKILL.md)...

## Other Notes:

[@poteto on Xitter:](https://x.com/poteto/status/2089067865098113024)

> every time you intervene and correct your agent, you should think about how to eliminate it entirely.
> 
> in order of value:
> 1. categorically eliminate the problem through better architecture or choice of data structures
> 2. turn it into a lint rule or test so CI catches it
> 3. turn it into a skill or rule
> 4. have humans review the code to catch it (ngmi)

---

Supposedly, Uncle Bob said something to the effect of "It's probably a mistake to impose human discipline on an agent. It is not a mistake to impose human values on the agent."

I can't verify this, but it is, coincidentally, roughly the idea behind subagent-driven development being codified in workflows instead of skills.
