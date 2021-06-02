---
title: "Understanding Allen Downey's Solution to the M&M Problem"
date: 2016-06-30T18:33:41+05:30
draft: false
tags: ["bayes", "probability"]
---

Allen Downey makes a very good case for learning advanced mathematics through
programming (Check the first section of the preface of _Think Bayes_, titled "My theory, which is mine").
But before the reader can hit paydirt with using the Bayes theorem in programming,
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
--------------

Simply put, it says that given data $D$ and a hypothesis $H$

$$ \begin{equation} P(H|D) = \frac{P(H)P(D|H)}{P(D)} \end{equation} $$

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

<table>
  <thead>
    <tr style="text-align: right;">
      <th>Color</th>
      <th>Before 1995</th>
      <th>After 1995</th>
     </tr>
  </thead>
  <tbody>
    <tr>
      <td>brown</td>
      <td>30 %</td>
      <td>13 %</td>
    </tr>
    <tr>
      <td>yellow</td>
      <td>20 %</td>
      <td>14 %</td>
    </tr>
    <tr>
      <td>red</td>
      <td>20 %</td>
      <td>13 %</td>
    </tr>
    <tr>
      <td>green</td>
      <td>10 %</td>
      <td>20 %</td>
    </tr>
    <tr>
      <td>orange</td>
      <td>10 %</td>
      <td>16 %</td>
    </tr>
    <tr>
      <td>tan</td>
      <td>10 %</td>
      <td>0 %</td>
    </tr>
    <tr>
      <td>blue</td>
      <td>0 %</td>
      <td>24 %</td>
    </tr>
  </tbody>
</table>

We have two bags full of M&Ms, one from 1994 and the other from 1996. We draw
an M&M each from the two bags, without knowing which M&M came from which bag.
One is yellow and one is green. The problem is:

> What is the probability that the yellow M&M came from the 1994 bag?

There can only be two hypotheses here:

* Hypothesis A ($H_{a}$): The yellow M&M came from the 1994 bag, and the green one came from the 1996 bag.
* Hypothesis B ($H_{b}$): The yellow M&M came from the 1996 bag, and the green one came from the 1994 bag.

Downey also introduces a useful notation for solving such problems, where we arrange the hypotheses and their corresponding Bayesian parameters in a table as follows:

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Prior</th>
      <th>Likelihood</th>
      <th>Normalizing Constant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>?</td>
      <td>?</td>
      <td>?</td>
    </tr>
    <tr>
      <th>B</th>
      <td>?</td>
      <td>?</td>
      <td>?</td>
    </tr>
  </tbody>
</table>

As we proceed with this problem, we'll also see how it serves as a good example of the diachronic interpretation of the Bayes' theorem - i.e., how it helps us update our belief about a given hypothesis once we've seen more data.


1. Our _priors_ about the two hypotheses would be naive, in that we would expect both hypotheses to be equiprobable, since we haven't considered the color distribution yet. Thus, $ P(H_{a}) = P(H_{b}) = \frac{1}{2}$
2. The likelihoods for the respective hypotheses can be readily obtained from the color distribution. Recall that $P(D|H_{a})$ is simply the probability that the observed data (one green, one yellow) is true for hypothesis $H_{a}$. Thus, $P(D|H_{a})$ equals the probability that the yellow M&M is from 1994 *and* the green one is from 1996. From the table, these values are both $\frac{1}{5}$. Thus, $P(D|H_{a}) = \frac{1}{25}$. Similarly, $P(D|H_{b}) = \frac{14}{100} \times \frac{10}{100} = \frac{7}{500}$.
3. Note that for each hypothesis, the product of the first two columns makes up the numerator of the right hand side of Bayes' equation.

So far so good, but I was stuck for a while before I could understand Downey's explanation of how he calculated the normalizing constant for the two cases. He writes that the third column is just the sum of the products of the first two columns. This means, that the normalizing constant is just the sum of the numerators for the respective Bayes' equations for the two scenarios. So,

$$ P(D) = P(H_{a})P(D|H_{a}) + P(H_{b})P(D|H_{b})$$

Thus,

$$ P(D) = \frac{1}{2} \times \frac{1}{25} + \frac{1}{2} \times \frac{7}{500} = \frac{27}{1000} $$

The catch is that this equation is perfectly in line with the second equation, subject to the assumption that both hypothesis are mutually exclusive (only one can be true) and collectively exhaustive (at least one must be true).

Since we now know all the values required to calculate the posteriors of both hypothesis, the rest is just arithmetic. It turns out that $P(H_{a}|D) = \frac{20}{27}$ and $P(H_{b}|D) = \frac{7}{27}$.

This was so much fun on paper, I can't wait to use these methods in Python. Needless to say I'm pretty excited about working through _Think Bayes_.
