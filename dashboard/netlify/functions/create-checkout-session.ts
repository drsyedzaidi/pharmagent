import Stripe from "stripe";
import { verifyUserToken } from "./_shared/verifyJwt";
import { supabaseAdmin } from "./_shared/supabaseAdmin";

export default async (req: Request) => {
  if (req.method !== "POST") return new Response("Method not allowed", { status: 405 });

  let userId: string;
  try {
    userId = await verifyUserToken(req.headers.get("authorization") ?? undefined);
  } catch (e) {
    return new Response(e instanceof Error ? e.message : "Unauthorized", { status: 401 });
  }

  const body = (await req.json().catch(() => null)) as { course_slug?: string } | null;
  const courseSlug = body?.course_slug;
  if (!courseSlug || typeof courseSlug !== "string") {
    return new Response("Missing course_slug", { status: 400 });
  }

  // Price always comes from the server-side courses table -- never trust a
  // client-supplied amount.
  const { data: course, error } = await supabaseAdmin()
    .from("courses")
    .select("slug, stripe_price_id, is_active")
    .eq("slug", courseSlug)
    .maybeSingle();

  if (error || !course || !course.is_active) {
    return new Response("Course not found", { status: 404 });
  }
  if (!course.stripe_price_id) {
    return new Response("This course isn't available for purchase yet.", { status: 409 });
  }

  const stripeSecret = process.env.STRIPE_SECRET_KEY;
  if (!stripeSecret) return new Response("Stripe is not configured.", { status: 500 });
  const stripe = new Stripe(stripeSecret);

  const origin = req.headers.get("origin") ?? new URL(req.url).origin;

  const session = await stripe.checkout.sessions.create({
    mode: "payment",
    line_items: [{ price: course.stripe_price_id, quantity: 1 }],
    metadata: { user_id: userId, course_slug: courseSlug },
    success_url: `${origin}/dashboard?checkout=success`,
    cancel_url: `${origin}/checkout/${courseSlug}`,
  });

  if (!session.url) return new Response("Could not create checkout session.", { status: 500 });
  return Response.json({ url: session.url });
};
