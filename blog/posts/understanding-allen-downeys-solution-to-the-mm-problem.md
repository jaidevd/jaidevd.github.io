<!-- 
.. title: Understanding Allen Downey's Solution to the M&M Problem
.. slug: understanding-allen-downeys-solution-to-the-mm-problem
.. date: 2016-06-29 20:13:56 UTC+05:30
.. tags: learning, bayes, programming, math, mathjax
.. category: 
.. link: 
.. description: 
.. type: text
-->

Allen Downey makes a very good case for learning advanced mathematics through
programming (Check the first section of the preface of _Think Bayes_, titled "My theory, which is mine").
But before can hit paydirt with using the Bayes theorem in programming,
Downey makes you go through some elementary problems in probability, which have
to be solved by hand first, if you expect to have a clear enough understanding
of the concept. I can vouch for this way of learning complex concepts. The way
I learnt the backpropagation algorithm (and its derivation), was with a pen,
paper and a calculator.
<!-- TEASER_END -->

Downey is also very careful about pointing out the difference between how
functions and operations manifest themselves in math and in programming. For
example, a function (say, $f(x)$) in mathematics can be implemented in software
by a number of things:

* An array, containing only data
* A routine that takes input(s) ($x$) and provides an output(s) ($f(x)$)
* A symbolic expression (commonly found in [CAS](https://en.wikipedia.org/wiki/Computer_algebra_system) libraries)

Not knowing these differences can severly handicap a programmer. I, for one,
found myself stymied multiple times in the very first chapter of _Think Bayes_,
even though I'm quite comfortable with what the Bayes theorem represents and
what it means for problems where belief or confidence needs to keep changing
with data. But here's the rub: I'm used to thinking about it very formally, in terms of
continuous functions, not discrete structures. And that has been the downfall
of many a programmer.

What follows is just narcissistic note-taking, which I hope won't let me forget
what I've already learnt.


Bayes' Theorem
==============

Simply put, it says that given data $D$ and a hypothesis $H$

$$ P(H|D) = \frac{P(H)P(D|H)}{P(D)} $$

where
$P(H)$ is the probability of the hypothesis, or the _prior_. $P(D|H)$ is the
_likelihood_, the probability that a favourable outcome (or anything that is
being observed) is true under the hypothesis $H$. $P(D)$ is the normalizing
constant, or the probability of the data being true in any case at all. The
significance of the normalizing constant is not immediately apparent, but we
can think of it as follows:

> If there are multiple possibilities, or multiple hypothesis, then $P(D)$ is
> the probability of an observation being true under any of them.

Concretely, this means that if there are $n$ hypotheses $H_{1}$ through
$H_{n}$, then

$$ P(D) = \sum_{i=1}^{n} P(H_{i}) P(D|H_{i}) $$


The M&M Problem
---------------

The problem states that the distribution of colors among M&Ms before and after
1995 is as follows:

|Color | Before 1995 | After 1995 |
|------|:-----------:|:----------:|
|brown |30 %         |13 %        |
|yellow|20 %         |14 %        |
|red   |20 %         |13 %        |
|green |10 %         |20 %        |
|orange|10 %         |16 %        |
|tan   |10 %         |0 %         |
|blue  |0 %          |24 %        |
